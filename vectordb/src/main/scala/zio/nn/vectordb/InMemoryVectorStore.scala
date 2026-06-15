package zio.nn.vectordb

import zio.nn.{VectorRecord, VectorStore}
import scala.util.Try
import scala.collection.concurrent.TrieMap

/** An in-memory [[VectorStore]] for testing and local use.
  *
  * Stores vectors in a concurrent hash map keyed by record id.
  * Search computes cosine distance against every stored vector — O(n)
  * per query. Not suitable for production-scale vector search.
  *
  * {{{
  *   val store = InMemoryVectorStore()
  *   store.store(VectorRecord("a", Array(1f, 0f))).get
  *   store.search(Array(1f, 0.1f), 1).get  // Seq(VectorRecord("a", ...))
  * }}}
  */
class InMemoryVectorStore extends VectorStore:

  private val records = TrieMap[String, VectorRecord]()

  override def store(record: VectorRecord): Try[Unit] = Try:
    records.put(record.id, record)

  override def storeBatch(recs: Seq[VectorRecord]): Try[Unit] = Try:
    recs.foreach(r => records.put(r.id, r))

  override def search(query: Array[Float], k: Int): Try[Seq[VectorRecord]] = Try:
    if records.isEmpty then Seq.empty
    else
      records.values.toSeq
        .map(r => (r, cosineDistance(query, r.values)))
        .sortBy(_._2)
        .take(k)
        .map(_._1)

  override def delete(id: String): Try[Unit] = Try:
    records.remove(id)

  override def deleteBatch(ids: Seq[String]): Try[Unit] = Try:
    ids.foreach(records.remove)

  override def close(): Unit = records.clear()

  /** List all stored records (for testing/inspection). */
  def listAll: Seq[VectorRecord] = records.values.toSeq

  /** Number of stored records. */
  def size: Int = records.size

  // ── helpers ─────────────────────────────────────────────────────────────

  private def cosineDistance(a: Array[Float], b: Array[Float]): Double =
    val dot = a.zip(b).map { case (x, y) => x * y }.sum
    val na  = math.sqrt(a.map(x => x * x).sum)
    val nb  = math.sqrt(b.map(x => x * x).sum)
    if na == 0 || nb == 0 then Double.MaxValue
    else 1.0 - (dot / (na * nb))

end InMemoryVectorStore

object InMemoryVectorStore:
  def apply(): InMemoryVectorStore = new InMemoryVectorStore()
