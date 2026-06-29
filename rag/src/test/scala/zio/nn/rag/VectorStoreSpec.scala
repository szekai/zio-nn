package zio.nn.rag

import zio.*
import zio.nn.VectorRecord
import zio.nn.vectordb.ZVectorStore
import zio.test.*
import zio.test.Assertion.*

object VectorStoreSpec extends ZIOSpecDefault:

  val doc1 = Document("1", "content one", Map("source" -> "test"), Array(0.1f, 0.2f, 0.3f))
  val doc2 = Document("2", "content two", Map("source" -> "test"), Array(0.4f, 0.5f, 0.6f))
  val doc3 = Document("3", "content three", Map("source" -> "other"), Array(0.7f, 0.8f, 0.9f))

  private val storeLayer: ULayer[ZVectorStore] = ZVectorStore.inMemory

  override def spec: Spec[TestEnvironment & Scope, Any] =
    suite("DocumentStore")(
      test("round-trip Document ↔ VectorRecord preserves all fields") {
        val doc = doc1
        val rec  = DocumentStore.toVectorRecord(doc)
        val back = DocumentStore.fromVectorRecord(rec)
        assertTrue(doc.id == back.id) &&
        assertTrue(doc.content == back.content) &&
        assertTrue(doc.metadata == back.metadata) &&
        assertTrue(doc.embedding.toList == back.embedding.toList)
      },
      test("inMemory ZVectorStore stores and searches documents") {
        for
          store <- ZIO.service[ZVectorStore]
          _     <- store.store(DocumentStore.toVectorRecord(doc1))
          _     <- store.store(DocumentStore.toVectorRecord(doc2))
          _     <- store.store(DocumentStore.toVectorRecord(doc3))
          hits  <- store.search(Array(0.1f, 0.2f, 0.3f), 2)
          ids   =  hits.map(_.id).toSet
        yield assertTrue(hits.nonEmpty) &&
              assertTrue(ids.contains("1"))
      },
      test("delete removes a document") {
        for
          store <- ZIO.service[ZVectorStore]
          _     <- store.store(DocumentStore.toVectorRecord(doc1))
          _     <- store.delete("1")
          hits  <- store.search(Array(0.1f, 0.2f, 0.3f), 5)
        yield assertTrue(hits.isEmpty)
      },
      test("search returns nearest neighbour sorted by distance") {
        for
          store <- ZIO.service[ZVectorStore]
          _     <- store.store(DocumentStore.toVectorRecord(doc1))
          _     <- store.store(DocumentStore.toVectorRecord(doc2))
          hits  <- store.search(Array(0.15f, 0.25f, 0.35f), 2)
          ids   =  hits.map(_.id)
        yield assertTrue(ids.headOption.contains("1"))
      }
    ).provideLayer(storeLayer)

end VectorStoreSpec
