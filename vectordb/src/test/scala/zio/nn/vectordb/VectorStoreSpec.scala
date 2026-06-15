package zio.nn.vectordb

import zio.*
import zio.test.*
import zio.test.Assertion.*
import zio.nn.VectorRecord

object VectorStoreSpec extends ZIOSpecDefault:

  private val vec1 = VectorRecord("a", Array(1f, 0f, 0f), Map("color" -> "red"))
  private val vec2 = VectorRecord("b", Array(0f, 1f, 0f), Map("color" -> "green"))
  private val vec3 = VectorRecord("c", Array(0f, 0f, 1f), Map("color" -> "blue"))
  private val all  = Seq(vec1, vec2, vec3)

  def spec = suite("VectorStore")(
    suite("Try-based API")(
      test("store and search returns nearest neighbours") {
        val store = InMemoryVectorStore()
        store.storeBatch(all).get
        val results = store.search(Array(0.9f, 0.1f, 0.0f), 2).get
        assertTrue(results.length == 2, results.head.id == "a")
      },

      test("delete removes a record") {
        val store = InMemoryVectorStore()
        store.storeBatch(all).get
        store.delete("a").get
        val results = store.search(Array(1f, 0f, 0f), 3).get
        assertTrue(results.length == 2, !results.exists(_.id == "a"), store.size == 2)
      },

      test("deleteBatch removes multiple records") {
        val store = InMemoryVectorStore()
        store.storeBatch(all).get
        store.deleteBatch(Seq("a", "b")).get
        assertTrue(store.size == 1, store.listAll.head.id == "c")
      },

      test("search on empty store returns empty") {
        val store = InMemoryVectorStore()
        val results = store.search(Array(1f, 0f, 0f), 5).get
        assertTrue(results.isEmpty)
      },

      test("close clears all records") {
        val store = InMemoryVectorStore()
        store.storeBatch(all).get
        store.close()
        assertTrue(store.size == 0)
      }
    ),

    suite("ZIO API")(
      test("storeZ and searchZ") {
        for
          store   <- ZIO.succeed(InMemoryVectorStore())
          _       <- store.storeBatchZ(all)
          results <- store.searchZ(Array(0.9f, 0.1f, 0.0f), 2)
        yield assertTrue(results.length == 2, results.head.id == "a")
      },

      test("storeZ, deleteZ, searchZ cycle") {
        for
          store <- ZIO.succeed(InMemoryVectorStore())
          _     <- store.storeZ(vec1)
          _     <- store.storeZ(vec2)
          all1  <- store.searchZ(Array(0f, 1f, 0f), 3)
          _     <- store.deleteZ("a")
          all2  <- store.searchZ(Array(0f, 1f, 0f), 3)
        yield assertTrue(all1.length == 2, all2.length == 1, all2.head.id == "b")
      },

      test("deleteBatchZ removes multiple records") {
        for
          store     <- ZIO.succeed(InMemoryVectorStore())
          _         <- store.storeBatchZ(all)
          _         <- store.deleteBatchZ(Seq("a", "c"))
          remaining <- store.searchZ(Array(0f, 1f, 0f), 3)
        yield assertTrue(remaining.length == 1, remaining.head.id == "b")
      }
    ),

    suite("Metadata")(
      test("vector records carry metadata through store/search") {
        val store = InMemoryVectorStore()
        val rec = VectorRecord("m1", Array(1f, 0f, 0f), Map("label" -> "test", "source" -> "spec"))
        store.store(rec).get
        val results = store.search(Array(1f, 0f, 0f), 1).get
        assertTrue(
          results.head.metadata.get("label").contains("test"),
          results.head.metadata.get("source").contains("spec")
        )
      }
    ),

    suite("Edge cases")(
      test("zero-length query returns all records") {
        val store = InMemoryVectorStore()
        store.storeBatch(all).get
        val results = store.search(Array.emptyFloatArray, 3).get
        assertTrue(results.length == 3)
      }
    )
  ) @@ TestAspect.sequential
end VectorStoreSpec
