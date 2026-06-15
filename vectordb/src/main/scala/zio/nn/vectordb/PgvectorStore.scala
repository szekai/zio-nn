package zio.nn.vectordb

import zio.nn.{VectorRecord, VectorStore}
import com.augustnagro.magnum.*
import com.pgvector.PGvector
import javax.sql.DataSource
import java.sql.{PreparedStatement, ResultSet, Types}
import scala.util.Try

/** A VectorStore implementation backed by PostgreSQL + pgvector.
  *
  * Uses Magnum for type-safe SQL and the pgvector JDBC driver for vector
  * similarity search.  All operations use `DataSource.getConnection`
  * internally (via Magnum's `Transactor` / `connect` / `transact`).
  *
  * The table `vector_embeddings` is created on construction with the
  * configured dimension.  PGvector's type mapping is registered on every
  * connection via the transactor's `connectionConfig` hook.
  *
  * @param ds
  *   the JDBC DataSource (connection pool or single connection)
  * @param dimension
  *   embedding dimension, default 768
  */
class PgvectorStore(ds: DataSource, val dimension: Int = 768) extends VectorStore:

  private val xa = Transactor(
    dataSource = ds,
    connectionConfig = conn => PGvector.addVectorType(conn)
  )

  private given DbCodec[PGvector] with
    val cols: IArray[Int]                          = IArray(Types.OTHER)
    val queryRepr: String                           = "?"
    def readSingle(rs: ResultSet, pos: Int): PGvector =
      rs.getObject(pos, classOf[PGvector])
    def readSingleOption(rs: ResultSet, pos: Int): Option[PGvector] =
      val res = rs.getObject(pos, classOf[PGvector])
      if rs.wasNull then None else Some(res)
    def writeSingle(v: PGvector, ps: PreparedStatement, pos: Int): Unit =
      ps.setObject(pos, v)

  // ── Initialisation: create the vector_embeddings table ────────────────
  initTable()
  private def initTable(): Unit =
    val conn = ds.getConnection
    try
      val stmt = conn.createStatement()
      stmt.execute(
        s"CREATE TABLE IF NOT EXISTS vector_embeddings (id TEXT PRIMARY KEY, embedding vector($dimension), metadata JSONB)"
      )
      stmt.close()
    finally conn.close()

  // ── VectorStore implementation ─────────────────────────────────────────

  override def store(record: VectorRecord): Try[Unit] = Try:
    transact(xa):
      val pgVec = PGvector(record.values)
      val meta  = metadataToJson(record.metadata)
      sql"""INSERT INTO vector_embeddings (id, embedding, metadata)
            VALUES (${record.id}, $pgVec, $meta::jsonb)
            ON CONFLICT (id) DO UPDATE SET
              embedding = EXCLUDED.embedding,
              metadata  = EXCLUDED.metadata""".update.run()

  override def storeBatch(records: Seq[VectorRecord]): Try[Unit] = Try:
    transact(xa):
      records.foreach { record =>
        val pgVec = PGvector(record.values)
        val meta  = metadataToJson(record.metadata)
        sql"""INSERT INTO vector_embeddings (id, embedding, metadata)
              VALUES (${record.id}, $pgVec, $meta::jsonb)
              ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                metadata  = EXCLUDED.metadata""".update.run()
      }

  override def search(query: Array[Float], k: Int): Try[Seq[VectorRecord]] = Try:
    val pgVec = PGvector(query)
    connect(xa):
      sql"""SELECT id, embedding, metadata
            FROM   vector_embeddings
            ORDER BY embedding <=> $pgVec
            LIMIT $k"""
        .query[(String, PGvector, String)]
        .run()
        .map { case (id, embedding, meta) =>
          VectorRecord(id, embedding.toArray, jsonToMetadata(meta))
        }

  override def delete(id: String): Try[Unit] = Try:
    transact(xa):
      sql"""DELETE FROM vector_embeddings WHERE id = $id""".update.run()

  override def deleteBatch(ids: Seq[String]): Try[Unit] = Try:
    transact(xa):
      ids.foreach { id =>
        sql"""DELETE FROM vector_embeddings WHERE id = $id""".update.run()
      }

  override def close(): Unit = ()

  // ── JSON helpers for metadata ──────────────────────────────────────────

  private def metadataToJson(m: Map[String, String]): String =
    if m.isEmpty then "{}"
    else
      m.map { case (k, v) =>
        s""""${escapeJson(k)}":"${escapeJson(v)}""""
      }.mkString("{", ",", "}")

  private def jsonToMetadata(json: String): Map[String, String] =
    if json == null || json.trim.isEmpty || json.trim == "{}" then Map.empty
    else
      val cleaned = json.trim.stripPrefix("{").stripSuffix("}")
      if cleaned.isBlank then Map.empty
      else
        cleaned.split(",").map(_.trim).flatMap { pair =>
          val colonIdx = pair.indexOf(":")
          if colonIdx > 0 then
            val key   = pair.substring(0, colonIdx).trim.stripPrefix("\"").stripSuffix("\"")
            val value = pair.substring(colonIdx + 1).trim.stripPrefix("\"").stripSuffix("\"")
            Some(key -> value)
          else None
        }.toMap

  private def escapeJson(s: String): String =
    s.replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")
      .replace("\t", "\\t")

end PgvectorStore
