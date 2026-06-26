package zio.nn.vectordb

import zio.*
import zio.nn.{VectorRecord, VectorStore}
import javax.sql.DataSource

/** ZIO-native vector store trait.
  *
  * Mirrors [[VectorStore]] with ZIO [[Task]] return types instead of
  * [[scala.util.Try]].  Provides managed lifecycle via [[ZLayer]] for both
  * [[InMemoryVectorStore]] and [[PgvectorStore]].
  *
  * ==Usage==
  *
  * Direct (in-memory):
  * {{{
  *   val layer: ULayer[ZVectorStore] = ZVectorStore.inMemory
  *   ZIO.serviceWithZIO[ZVectorStore](_.storeZ(myRecords))
  * }}}
  *
  * Direct (pgvector — managed JDBC):
  * {{{
  *   val layer: TaskLayer[ZVectorStore] = ZVectorStore.pgvector(ds)
  *   ZIO.serviceWithZIO[ZVectorStore](_.searchZ(query, 5))
  * }}}
  *
  * Wrap existing [[VectorStore]]:
  * {{{
  *   val store: VectorStore = ???
  *   val zStore: ZVectorStore = ZVectorStore.fromVectorStore(store)
  * }}}
  */
trait ZVectorStore:

  /** Store a single embedding record. */
  def store(record: VectorRecord): Task[Unit]

  /** Store multiple records in a single batch. */
  def storeBatch(records: Seq[VectorRecord]): Task[Unit]

  /** Search for the `k` nearest neighbours of `query`.
    *
    * @param query  the query embedding vector
    * @param k      number of neighbours to return
    * @return       top-k records ordered by ascending distance
    */
  def search(query: Array[Float], k: Int): Task[Seq[VectorRecord]]

  /** Delete a single record by id. */
  def delete(id: String): Task[Unit]

  /** Delete multiple records by ids. */
  def deleteBatch(ids: Seq[String]): Task[Unit]

end ZVectorStore


object ZVectorStore:

  /** Wrap a [[VectorStore]] into a [[ZVectorStore]].
    *
    * Each method delegates to the underlying store, wrapping the `Try` in a
    * `ZIO.attemptBlocking`.  The lifecycle of the underlying store (e.g.
    * connection pool close) is managed externally — the returned
    * [[ZVectorStore]] does not own it.
    *
    * Use this to integrate any [[VectorStore]] (including third-party
    * implementations) into ZIO-based applications.
    */
  def fromVectorStore(underlying: VectorStore): ZVectorStore =
    new ZVectorStore:
      def store(record: VectorRecord): Task[Unit] =
        ZIO.attemptBlocking(underlying.store(record).get)

      def storeBatch(records: Seq[VectorRecord]): Task[Unit] =
        ZIO.attemptBlocking(underlying.storeBatch(records).get)

      def search(query: Array[Float], k: Int): Task[Seq[VectorRecord]] =
        ZIO.attemptBlocking(underlying.search(query, k).get)

      def delete(id: String): Task[Unit] =
        ZIO.attemptBlocking(underlying.delete(id).get)

      def deleteBatch(ids: Seq[String]): Task[Unit] =
        ZIO.attemptBlocking(underlying.deleteBatch(ids).get)

  // ── ZLayer helpers ──────────────────────────────────────────────────────

  /** [[ULayer]] backed by a fresh [[InMemoryVectorStore]].
    *
    * No cleanup is required — the in-memory store is garbage-collected.
    *
    * {{{
    *   val layer: ULayer[ZVectorStore] = ZVectorStore.inMemory
    * }}}
    */
  val inMemory: ULayer[ZVectorStore] =
    ZLayer.succeed(fromVectorStore(InMemoryVectorStore()))

  /** [[ZLayer]] backed by [[PgvectorStore]] with managed lifecycle.
    *
    * The underlying JDBC connection and pgvector table are acquired on
    * layer creation and released when the layer's [[Scope]] ends.
    *
    * @param ds         JDBC DataSource (connection pool or single connection)
    * @param dimension  embedding dimension, default 768
    *
    * {{{
    *   val layer: TaskLayer[ZVectorStore] = ZVectorStore.pgvector(ds, 384)
    * }}}
    */
  def pgvector(
    ds: DataSource,
    dimension: Int = 768
  ): ZLayer[Any, Throwable, ZVectorStore] =
    ZLayer.scoped:
      for
        pgStore <- ZIO.acquireRelease(
          ZIO.attemptBlocking(PgvectorStore(ds, dimension))
        )(store => ZIO.attemptBlocking(store.close()).orDie)
      yield fromVectorStore(pgStore)

  /** Variant of [[pgvector]] that reads the dimension from ZIO Config.
    *
    * {{{
    *   given Config[Int] = Config.integer("embedding.dimension").withDefault(768)
    *   val layer: TaskLayer[ZVectorStore] = ZVectorStore.pgvector(ds)
    * }}}
    */
  def pgvector(
    ds: DataSource
  )(using config: Config[Int]): ZLayer[Any, Throwable, ZVectorStore] =
    ZLayer.scoped:
      for
        dim     <- ZIO.config[Int]
        pgStore <- ZIO.acquireRelease(
          ZIO.attemptBlocking(PgvectorStore(ds, dim))
        )(store => ZIO.attemptBlocking(store.close()).orDie)
      yield fromVectorStore(pgStore)

end ZVectorStore
