package zio.nn.rag

import zio.*
import zio.nn.VectorRecord
import zio.nn.vectordb.ZVectorStore

/** A document with text content, optional metadata, and an optional embedding vector.
  *
  * Documents are the fundamental unit in the RAG pipeline:
  *   - '''ingest''': a `Document` is created, optionally embedded, and stored.
  *   - '''query''': a text query is embedded and compared against stored documents
  *     via vector similarity search.
  *
  * @param id        unique document identifier
  * @param content   the raw text content
  * @param metadata  key-value metadata (e.g. source, date, tags)
  * @param embedding pre-computed embedding vector (empty if not yet embedded)
  */
case class Document(
  id: String,
  content: String,
  metadata: Map[String, String] = Map.empty,
  embedding: Array[Float] = Array.emptyFloatArray
)

/** Conversion helpers and ZLayer factories that bridge the `Document` model
  * with `zio.nn.vectordb.ZVectorStore` / `zio.nn.VectorRecord`.
  *
  * The internal `__content` metadata key stores the raw text so that
  * round-trips through [[ZVectorStore]] preserve the content field.
  */
object DocumentStore:
  private val ContentKey = "__content"

  /** Convert a [[Document]] to a [[VectorRecord]] for storage.
    * Content is stashed in the record's `metadata` map under `__content`.
    */
  def toVectorRecord(doc: Document): VectorRecord =
    VectorRecord(doc.id, doc.embedding, doc.metadata + (ContentKey -> doc.content))

  /** Reconstruct a [[Document]] from a [[VectorRecord]].
    * Content is extracted from the `__content` metadata key.
    */
  def fromVectorRecord(rec: VectorRecord): Document =
    Document(
      id       = rec.id,
      content  = rec.metadata.getOrElse(ContentKey, ""),
      metadata = rec.metadata - ContentKey,
      embedding = rec.values
    )

  /** In-memory vector store layer — useful for testing or single-JVM apps. */
  val inMemory: ZLayer[Any, Nothing, ZVectorStore] = ZVectorStore.inMemory
