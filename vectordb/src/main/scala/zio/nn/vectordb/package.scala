package zio.nn

import zio.*
import zio.stream.*
import javax.sql.DataSource

/** ZIO-friendly wrappers for [[VectorStore]] and vector DB resource management.
  *
  * Provides extension methods on [[VectorStore]] that wrap every Try-based
  * operation in ZIO, plus managed resource creation for [[PgvectorStore]]
  * and [[ZVectorStore]] for fully ZIO-native access.
  *
  * == Quick comparison ==
  *
  * | Approach | Trait | Return type | Lifecycle |
  * |----------|-------|-------------|-----------|
  * | Try-based | [[VectorStore]] | `Try[A]` | manual `close()` |
  * | Extension methods | `store.storeZ(...)` | `Task[A]` | manual `close()` |
  * | ZIO-native | [[ZVectorStore]] | `Task[A]` | [[ZLayer]] (scoped) |
  *
  * === Extension methods (backward-compatible) ===
  * {{{
  *   import zio.nn.vectordb.*
  *
  *   val store: VectorStore = ???
  *   store.storeZ(VectorRecord("a", Array(1f, 0f)))    // Task[Unit]
  *   store.searchZ(Array(1f, 0.1f), 3)                  // Task[Seq[VectorRecord]]
  * }}}
  *
  * === ZVectorStore via ZLayer ===
  * {{{
  *   // In-memory (testing / single-node)
  *   val layer: ULayer[ZVectorStore] = ZVectorStore.inMemory
  *
  *   // Postgres + pgvector (production)
  *   val layer: TaskLayer[ZVectorStore] = ZVectorStore.pgvector(dataSource, 768)
  * }}}
  *
  * === Managed PgvectorStore (direct, without ZVectorStore) ===
  * {{{
  *   ZIO.scoped {
  *     createPgvectorStore(ds).flatMap { store =>
  *       store.storeZ(record) *> store.searchZ(query, 5)
  *     }
  *   }
  * }}}
  */
package object vectordb:

  // ── ZIO extensions on VectorStore ─────────────────────────────────────

  extension (store: VectorStore)

    /** Store a single embedding record. */
    def storeZ(record: VectorRecord): Task[Unit] =
      ZIO.attemptBlocking(store.store(record).get)

    /** Store multiple records in a single batch. */
    def storeBatchZ(records: Seq[VectorRecord]): Task[Unit] =
      ZIO.attemptBlocking(store.storeBatch(records).get)

    /** Search for the `k` nearest neighbours of `query`. */
    def searchZ(query: Array[Float], k: Int): Task[Seq[VectorRecord]] =
      ZIO.attemptBlocking(store.search(query, k).get)

    /** Delete a single record by id. */
    def deleteZ(id: String): Task[Unit] =
      ZIO.attemptBlocking(store.delete(id).get)

    /** Delete multiple records by ids. */
    def deleteBatchZ(ids: Seq[String]): Task[Unit] =
      ZIO.attemptBlocking(store.deleteBatch(ids).get)

  // ── Managed resource: PgvectorStore ──────────────────────────────────

  /** Create a [[PgvectorStore]] as a ZIO managed resource.
    *
    * The underlying [[PgvectorStore]] is closed when the [[Scope]] ends.
    *
    * @param ds         JDBC DataSource (connection pool or single connection)
    * @param dimension  embedding dimension, default 768
    */
  def createPgvectorStore(
    ds: DataSource,
    dimension: Int = 768
  ): ZIO[Scope, Throwable, PgvectorStore] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(PgvectorStore(ds, dimension))
    )(store => ZIO.attemptBlocking(store.close()).orDie)

  /** Convenience variant that reads dimension from ZIO Config. */
  def createPgvectorStore(
    ds: DataSource
  )(using Config[Int]): ZIO[Scope, Throwable, PgvectorStore] =
    ZIO.config[Int].flatMap(dim => createPgvectorStore(ds, dim))

end vectordb
