package zio.nn.djl

import zio.*
import zio.stream.*
import zio.nn.FitResult
import java.nio.file.Path

object zioApi:

  extension (model: ZModel)
    def predictZ(features: Array[Array[Float]]): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predict(features).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def predictDoubleZ(features: Array[Array[Double]]): Task[Array[Double]] =
      ZIO.attemptBlocking { val f = features.map(_.map(_.toFloat)); model.predict(f).get.map(_.toDouble) }

    def fitDoubleZ(features: Array[Array[Double]], labels: Array[Double], epochs: Int, lr: Double = 0.001): Task[FitResult] =
      ZIO.attemptBlocking { val f = features.map(_.map(_.toFloat)); val l = labels.map(_.toFloat); model.fit(f, l, epochs, lr.toFloat).get }

    def predictFlow: ZPipeline[Any, Throwable, Array[Array[Float]], Array[Float]] =
      ZPipeline.mapZIO(features => predictZ(features))

    def fitFlow(epochs: Int = 1, lr: Float = 0.001f): ZPipeline[Any, Throwable, (Array[Array[Float]], Array[Float]), FitResult] =
      ZPipeline.mapZIO((feats, labels) => fitZ(feats, labels, epochs, lr))

    def predictTimed(features: Array[Array[Float]]): ZIO[Any, Throwable, Array[Float]] =
      predictZ(features).timed.flatMap((duration, result) => ZIO.logDebug(s"predict: ${duration.toMillis}ms").as(result))

    def fitTimed(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): ZIO[Any, Throwable, FitResult] =
      fitZ(features, labels, epochs, lr).timed.flatMap((duration, result) => ZIO.logInfo(s"fit($epochs epochs): ${duration.toMillis}ms, loss=${result.loss}").as(result))

    def fitWithCheckpoints(features: Array[Array[Float]], labels: Array[Float], epochs: Int, saveEvery: Int, checkpointPath: String, lr: Float = 0.001f): ZIO[Any, Throwable, FitResult] =
      ZIO.suspendSucceed {
        def loop(epoch: Int): ZIO[Any, Throwable, FitResult] =
          if epoch > epochs then ZIO.succeed(FitResult(Double.NaN, 0))
          else fitZ(features, labels, saveEvery, lr) *>
            ZIO.attemptBlocking(model.save(Path.of(s"$checkpointPath-epoch$epoch"))).ignore *>
            ZIO.logInfo(s"Checkpoint saved at epoch $epoch") *> loop(epoch + saveEvery)
        loop(1)
      }

  def create(arch: zio.nn.ModelDef, name: String = "model", engine: String = "PyTorch"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(ZIO.attemptBlocking(ZModel.create(arch, name, engine).get))(m => ZIO.attemptBlocking(m.close()).orDie)

  def load(path: Path, name: String = "model", engine: String = "PyTorch"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(ZIO.attemptBlocking(ZModel.load(path, name, engine).get))(m => ZIO.attemptBlocking(m.close()).orDie)
