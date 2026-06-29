package zio.nn.rag

import zio.*

/** Adapter that exposes RAG lookups as a string-returning interface for AI agent tool calls.
  *
  * This bridge does not own a VectorStore — it delegates all storage and
  * retrieval through [[RAGService]], whose layer provides [[ZVectorStore]] internally.
  *
  * The `search` method returns a human-readable string of top-K documents
  * (ID + content blocks joined by `---` separators), suitable for passing
  * directly into an LLM prompt as context.
  *
  * Score information is intentionally omitted (not yet surfaced from the vector
  * store backend); the caller treats returned documents as equally ranked.
  */
trait RAGAgentBridge:
  /** Search for documents relevant to `query` and return a formatted string.
    *
    * @param query the search query text
    * @param topK  number of documents to retrieve (default 5)
    * @return a newline-separated block of `[id]\ncontent` entries
    */
  def search(query: String, topK: Int): Task[String]
end RAGAgentBridge

/** Default [[RAGAgentBridge]] implementation.
  *
  * @param rag the underlying RAG service used for query execution
  */
case class RAGAgentBridgeLive(rag: RAGService) extends RAGAgentBridge:
  override def search(query: String, topK: Int): Task[String] =
    rag.query(query, topK).map { docs =>
      docs
        .map(d => s"[${d.id}] (score: N/A)\n${d.content}")
        .mkString("\n---\n")
    }
end RAGAgentBridgeLive

object RAGAgentBridge:
  /** ZLayer that wires [[RAGService]] into [[RAGAgentBridge]]. */
  val layer: ZLayer[RAGService, Nothing, RAGAgentBridge] =
    ZLayer.fromFunction(RAGAgentBridgeLive(_))
end RAGAgentBridge
