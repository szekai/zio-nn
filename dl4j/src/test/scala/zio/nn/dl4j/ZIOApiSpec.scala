package zio.nn.dl4j

import zio.nn.*
import zio.nn.dsl.*
import zio.test.*
import zio.test.Assertion.*
import zio.*
import zio.nn.dl4j.zioApi.*

object ZIOApiSpec extends ZIOSpecDefault:

  def spec = suite("ZIO API (DL4J)")(
    test("create returns Scoped ZModel") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      for
        model <- zioApi.create(arch)
        _     <- ZIO.attemptBlocking(model.close()).orDie
      yield assertTrue(true)
    },
    test("predictZ works") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          result <- model.predictZ(Array.fill(2)(Array.fill(7)(0.5f)))
          _      <- ZIO.attemptBlocking(model.close()).orDie
        yield assertTrue(result.length == 2)
      }
    },
    test("predictDoubleZ works with Double precision") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          result <- model.predictDoubleZ(Array.fill(1)(Array.fill(7)(0.5)))
          _      <- ZIO.attemptBlocking(model.close()).orDie
        yield assertTrue(result.length == 1)
      }
    },
    test("fitWithCheckpoints returns real epoch count across partial chunks") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          tmpDir <- ZIO.attemptBlocking(java.nio.file.Files.createTempDirectory("zionn-dl4j-checkpoints"))
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
    },
    test("fit() returns lossHistory with per-epoch values") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      val model = ZModel.create(arch)
      val feats = Array.fill(6)(Array.fill(7)(scala.util.Random.nextFloat()))
      val labels = Array.fill(6)(scala.util.Random.nextFloat())
      val result = model.fit(feats, labels, 3)
      model.close()
      assertTrue(
        result.isSuccess,
        result.get.lossHistory.length == 3,
        result.get.lossHistory.forall(!_.isNaN)
      )
    },
    test("fitWithCallbacksZ fires EpochEnd events") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      var epochCount = 0
      val listener = new TrainingCallback:
        def onEvent(event: TrainingEvent) = event match
          case TrainingEvent.EpochEnd(_, _, _) => ZIO.succeed { epochCount += 1 }
          case _ => ZIO.unit
      zioApi.create(arch).flatMap { model =>
        model.fitWithCallbacksZ(
          features = Array.fill(4)(Array.fill(7)(0.25f)),
          labels = Array.fill(4)(0.5f),
          epochs = 3,
          callbacks = List(listener)
        ).map { result =>
          model.close()
          assertTrue(epochCount == 3, result.lossHistory.length == 3)
        }
      }
    },
    test("fitWithCallbacksZ fires TrainEnd event") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      var trainEnded = false
      val listener = new TrainingCallback:
        def onEvent(event: TrainingEvent) = event match
          case TrainingEvent.TrainEnd(_) => ZIO.succeed { trainEnded = true }
          case _ => ZIO.unit
      zioApi.create(arch).flatMap { model =>
        model.fitWithCallbacksZ(
          features = Array.fill(4)(Array.fill(7)(0.25f)),
          labels = Array.fill(4)(0.5f),
          epochs = 2,
          callbacks = List(listener)
        ).as {
          model.close()
          assertTrue(trainEnded)
        }
      }
    }
  ).provideLayer(Scope.default)
