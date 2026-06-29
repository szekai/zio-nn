package zio.nn.rag

import zio.*
import zio.nn.VectorRecord
import zio.nn.vectordb.ZVectorStore
import zio.test.*
import zio.test.Assertion.*

object RAGServiceSpec extends ZIOSpecDefault:

  /** Mock embedder that returns a constant Array[Float] padded to dimension 3. */
  class ConstantEmbedder extends EmbeddingService:
    private val dim = 3
    override def embed(text: String): Task[Array[Float]] =
      ZIO.succeed(Array.tabulate(dim)(i => (text.hashCode * (i + 1)) % 100).map(_.toFloat.abs))
    override def embedBatch(texts: Chunk[String]): Task[Chunk[Array[Float]]] =
      ZIO.foreach(texts)(embed)

  val embedder = new ConstantEmbedder

  val doc1 = Document("1", "apple banana",  Map("tag" -> "fruit"),   Array(1f, 0f, 0f))
  val doc2 = Document("2", "carrot daikon", Map("tag" -> "veg"),     Array(0f, 1f, 0f))
  val doc3 = Document("3", "egg frittata",  Map("tag" -> "protein"), Array(0f, 0f, 1f))

  override def spec: Spec[TestEnvironment & Scope, Any] =
    suite("RAGService")(
      test("ingest and query with pre-embedded documents") {
        for
          store <- ZIO.service[ZVectorStore]
          rag   =  RAGServiceLive(store, new ConstantEmbedder)
          emb   <- embedder.embed("apple banana")
          doc1  =  Document("1", "apple banana", Map("tag" -> "fruit"), emb)
          _     <- rag.ingest(doc1)
          _     <- rag.ingest(doc2)
          _     <- rag.ingest(doc3)
          hits  <- rag.query("apple banana", 3)
        yield assertTrue(hits.exists(_.id == "1"))
      },
      test("ingest with auto-embedding (empty embedding)") {
        for
          store <- ZIO.service[ZVectorStore]
          rag   =  RAGServiceLive(store, new ConstantEmbedder)
          raw   =  Document("42", "something entirely new", Map.empty, Array.emptyFloatArray)
          _     <- rag.ingest(raw)
          hits  <- rag.query("something", 3)
        yield assertTrue(hits.exists(_.id == "42"))
      },
      test("delete removes document from store") {
        for
          store <- ZIO.service[ZVectorStore]
          rag   =  RAGServiceLive(store, new ConstantEmbedder)
          _     <- rag.ingest(doc1)
          _     <- rag.deleteDocument("1")
          hits  <- rag.query("apple", 5)
        yield assertTrue(!hits.exists(_.id == "1"))
      },
      test("ingest batch inserts multiple documents") {
        for
          store <- ZIO.service[ZVectorStore]
          rag   =  RAGServiceLive(store, new ConstantEmbedder)
          _     <- rag.ingestBatch(Chunk(doc1, doc2, doc3))
          hits  <- rag.query("", 10)
        yield assertTrue(hits.size >= 3)
      }
    ).provideLayer(ZVectorStore.inMemory)

end RAGServiceSpec
