package zio.nn.djl

import zio.*
import zio.nn.dsl.*
import zio.nn.djl.zioApi.*
import zio.test.*
import zio.test.Assertion.*

object ZIOApiSpec extends ZIOSpecDefault:

  def spec = suite("ZIO API (DJL)")(
    test("create returns Scoped ZModel") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      for
        model <- zioApi.create(arch)
        _     <- ZIO.attemptBlocking(model.close()).orDie
      yield assertTrue(true)
    },
    test("fitWithCheckpoints returns real epoch count across partial chunks") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          tmpDir <- ZIO.attemptBlocking(java.nio.file.Files.createTempDirectory("zionn-djl-checkpoints"))
          result <- model.fitWithCheckpoints(
                      features = Array.fill(4)(Array.fill(7)(0.25f)),
                      labels = Array.fill(4)(0.5f),
                      epochs = 3,
                      saveEvery = 2,
                      checkpointPath = tmpDir.resolve("checkpoint").toString
                    )
          _      <- ZIO.attemptBlocking(model.close()).orDie
        yield assertTrue(result.epochs == 3, !result.loss.isNaN)
      }
    },
    test("fitWithCheckpoints rejects non-positive saveEvery") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          exit <- model.fitWithCheckpoints(
                    features = Array.fill(2)(Array.fill(7)(0.25f)),
                    labels = Array.fill(2)(0.5f),
                    epochs = 2,
                    saveEvery = 0,
                    checkpointPath = "ignored"
                  ).exit
          _    <- ZIO.attemptBlocking(model.close()).orDie
        yield assert(exit)(fails(isSubtype[IllegalArgumentException](anything)))
      }
    }
  ).provideLayer(Scope.default)
