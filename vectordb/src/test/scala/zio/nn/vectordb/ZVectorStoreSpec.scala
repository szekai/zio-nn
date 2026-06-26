package zio.nn.vectordb

import zio.*
import zio.test.*
import zio.test.Assertion.*
import zio.nn.VectorRecord

object ZVectorStoreSpec extends ZIOSpecDefault:

  private val vec1 = VectorRecord("a", Array(1f, 0f, 0f), Map("color" -> "red"))
  private val vec2 = VectorRecord("b", Array(0f, 1f, 0f), Map("color" -> "green"))
  private val vec3 = VectorRecord("c", Array(0f, 0f, 1f), Map("color" -> "blue"))
  private val all  = Seq(vec1, vec2, vec3)

  def spec = suite("ZVectorStore")(
    suite("inMemory layer")(
      test("store and search returns nearest neighbours") {
        for
          store   <- ZIO.service[ZVectorStore]
          _       <- store.storeBatch(all)
          results <- store.search(Array(0.9f, 0.1f, 0.0f), 2)
        yield assertTrue(results.length == 2, results.head.id == "a")
      },

      test("delete removes a record") {
        for
          store   <- ZIO.service[ZVectorStore]
          _       <- store.storeBatch(all)
          _       <- store.delete("a")
          results <- store.search(Array(1f, 0f, 0f), 3)
        yield assertTrue(results.length == 2, !results.exists(_.id == "a"))
      },

      test("deleteBatch removes multiple records") {
        for
          store     <- ZIO.service[ZVectorStore]
          _         <- store.storeBatch(all)
          _         <- store.deleteBatch(Seq("a", "b"))
          remaining <- store.search(Array(0f, 1f, 0f), 3)
        yield assertTrue(remaining.length == 1, remaining.head.id == "c")
      },

      test("search on empty store returns empty") {
        for
          store   <- ZIO.service[ZVectorStore]
          results <- store.search(Array(1f, 0f, 0f), 5)
        yield assertTrue(results.isEmpty)
      }
    ),

    suite("fromVectorStore wrapper")(
      test("wraps InMemoryVectorStore correctly") {
        val raw    = InMemoryVectorStore()
        val zStore = ZVectorStore.fromVectorStore(raw)
        for
          _       <- zStore.storeBatch(all)
          results <- zStore.search(Array(0f, 1f, 0f), 1)
        yield assertTrue(results.head.id == "b")
      },

      test("deleteBatchZ via wrapper") {
        val raw    = InMemoryVectorStore()
        val zStore = ZVectorStore.fromVectorStore(raw)
        for
          _         <- zStore.storeBatch(all)
          _         <- zStore.deleteBatch(Seq("a", "c"))
          remaining <- zStore.search(Array(0f, 1f, 0f), 3)
        yield assertTrue(remaining.length == 1, remaining.head.id == "b")
      }
    ),

    suite("Metadata")(
      test("records carry metadata through store/search") {
        for
          store   <- ZIO.service[ZVectorStore]
          rec      = VectorRecord("m1", Array(1f, 0f, 0f), Map("label" -> "test", "source" -> "spec"))
          _       <- store.store(rec)
          results <- store.search(Array(1f, 0f, 0f), 1)
        yield assertTrue(
          results.head.metadata.get("label").contains("test"),
          results.head.metadata.get("source").contains("spec")
        )
      }
    ),

    suite("Edge cases")(
      test("zero-length query returns all records") {
        for
          store   <- ZIO.service[ZVectorStore]
          _       <- store.storeBatch(all)
          results <- store.search(Array.emptyFloatArray, 3)
        yield assertTrue(results.length == 3)
      },

      test("store/delete cycle on same id") {
        for
          store   <- ZIO.service[ZVectorStore]
          _       <- store.store(vec1)
          _       <- store.store(vec1)   // overwrite
          _       <- store.delete("a")
          results <- store.search(Array(1f, 0f, 0f), 3)
        yield assertTrue(results.isEmpty)
      }
    )
  ).provideLayer(ZVectorStore.inMemory) @@ TestAspect.sequential
end ZVectorStoreSpec
