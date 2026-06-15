package zio.nn.dl4j

import zio.nn.*
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.*
import zio.nn.vectordb.InMemoryVectorStore
import zio.test.*
import zio.test.Assertion.*
import zio.*
import zio.stream.*

object EgressSpec extends ZIOSpecDefault:

  def spec = suite("Egress (predictAndStoreFlow)")(
    test("predictAndStoreFlow stores predictions and returns them") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          store   <- ZIO.succeed(InMemoryVectorStore())
          stream   = ZStream(
                       (Array(Array.fill(7)(0.5f), Array.fill(7)(0.3f)), Array("a", "b")),
                       (Array(Array.fill(7)(0.9f)), Array("c"))
                     )
          batchPredictions <- stream.via(model.predictAndStoreFlow(store)).runCollect
          stored            = store.listAll
        yield
          model.close()
          assertTrue(
            batchPredictions.length == 2,
            stored.length == 3,
            stored.map(_.id).toSet == Set("a", "b", "c"),
            stored.forall(_.values.length == 1)
          )
      }
    },

    test("predictAndStore returns predictions") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          store   <- ZIO.succeed(InMemoryVectorStore())
          result  <- model.predictAndStoreZ(
                       Array(Array.fill(7)(0.4f), Array.fill(7)(0.6f)),
                       store,
                       Array("x", "y")
                     )
          stored   = store.listAll
        yield
          model.close()
          assertTrue(
            result.length == 2,
            stored.length == 2,
            stored.map(_.id).toSet == Set("x", "y")
          )
      }
    }
  ).provideLayer(Scope.default) @@ TestAspect.sequential
end EgressSpec
