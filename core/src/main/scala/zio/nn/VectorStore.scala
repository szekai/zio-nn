package zio.nn

import scala.util.Try

/** A vector embedding record to be stored in a vector database.
  *
  * @param id       unique identifier for the vector
  * @param values   the embedding vector (float array)
  * @param metadata optional key-value pairs for filtering / tags
  */
case class VectorRecord(
  id: String,
  values: Array[Float],
  metadata: Map[String, String] = Map.empty
)

/** Backend-agnostic interface for storing and querying vector embeddings.
  *
  * The trait lives in `core/` so that any backend module can depend on the
  * interface without pulling in database drivers.  Concrete implementations
  * (e.g. pgvector) live in separated modules under `vectordb/`.
  *
  * All methods return `Try` for synchronous use.  ZIO wrappers live in
  * each backend's `zioApi` package object.
  */
trait VectorStore:

  /** Store a single embedding record. */
  def store(record: VectorRecord): Try[Unit]

  /** Store multiple records in a single batch. */
  def storeBatch(records: Seq[VectorRecord]): Try[Unit]

  /** Search for the `k` nearest neighbours of `query`.
    *
    * @param query  the query embedding vector
    * @param k      number of neighbours to return
    * @return       top-k records ordered by ascending distance
    */
  def search(query: Array[Float], k: Int): Try[Seq[VectorRecord]]

  /** Delete a single record by id. */
  def delete(id: String): Try[Unit]

  /** Delete multiple records by ids. */
  def deleteBatch(ids: Seq[String]): Try[Unit]

  /** Release underlying resources (connection pool, etc.). */
  def close(): Unit
