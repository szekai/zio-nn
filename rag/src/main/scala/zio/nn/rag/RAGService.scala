package zio.nn.rag

import zio.*
import zio.nn.VectorRecord
import zio.nn.vectordb.ZVectorStore

/** Core RAG pipeline that combines a vector store with an embedding service.
  *
  * The RAG (Retrieval-Augmented Generation) flow:
  *   1. '''ingest''': documents are embedded (if not already), converted to
  *      [[VectorRecord]]s, and stored in [[ZVectorStore]].
  *   2. '''query''': a question is embedded, the vector store is searched for
  *      the top-K nearest neighbors, and the matching [[Document]]s are returned.
  *   3. '''delete''': documents are removed by ID from the store.
  */
trait RAGService:
  /** Ingest a single document — embeds content if not already embedded. */
  def ingest(document: Document): Task[Unit]

  /** Ingest a batch of documents in a single embedding + store operation. */
  def ingestBatch(documents: Chunk[Document]): Task[Unit]

  /** Query the store for documents similar to a given question.
    *
    * @param question the query text
    * @param topK     number of nearest neighbors to return (default 5)
    */
  def query(question: String, topK: Int = 5): Task[Chunk[Document]]

  /** Delete a document from the store by its ID. */
  def deleteDocument(id: String): Task[Unit]
end RAGService

/** Default [[RAGService]] implementation backed by a [[ZVectorStore]] and [[EmbeddingService]].
  *
  * For documents that already have a non-empty embedding vector, the embed
  * step is skipped (idempotent re-ingest support).
  *
  * @param store    the backing vector store
  * @param embedder the embedding service
  */
case class RAGServiceLive(store: ZVectorStore, embedder: EmbeddingService) extends RAGService:

  override def ingest(document: Document): Task[Unit] =
    for
      enriched <- embed(document)
      record   =  DocumentStore.toVectorRecord(enriched)
      _        <- store.store(record)
    yield ()

  override def ingestBatch(documents: Chunk[Document]): Task[Unit] =
    for
      enriched <- ZIO.foreach(documents)(embed)
      records  =  enriched.map(DocumentStore.toVectorRecord)
      _        <- store.storeBatch(records)
    yield ()

  override def query(question: String, topK: Int): Task[Chunk[Document]] =
    for
      qEmbed  <- embedder.embed(question)
      results <- store.search(qEmbed, topK)
    yield Chunk.fromIterable(results.map(DocumentStore.fromVectorRecord))

  override def deleteDocument(id: String): Task[Unit] =
    store.delete(id)

  private def embed(doc: Document): Task[Document] =
    if doc.embedding.nonEmpty then ZIO.succeed(doc)
    else embedder.embed(doc.content).map(emb => doc.copy(embedding = emb))
end RAGServiceLive

object RAGService:
  /** ZLayer that wires a [[ZVectorStore]] and [[EmbeddingService]] into [[RAGService]]. */
  val live: ZLayer[ZVectorStore & EmbeddingService, Nothing, RAGService] =
    ZLayer.fromFunction(RAGServiceLive(_, _))
end RAGService
