package zio.nn.dl4j

import zio.*
import zio.stream.*
import zio.nn.FitResult
import java.io.File

/** ZIO-native API — Task-based with Scope resource management and ZStream support.
  *
  * Usage:
  * {{{
  *   import zio.nn.dl4j.zioApi.*
  *   create(arch).flatMap(_.predictZ(features))
  *   featureStream.via(model.predictFlow)
  * }}}
  */
object zioApi:

  extension (model: ZModel)
    def predictZ(features: Array[Array[Float]]): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predict(features).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def predictDoubleZ(features: Array[Array[Double]]): Task[Array[Double]] =
      ZIO.attemptBlocking {
        val f = features.map(_.map(_.toFloat))
        model.predict(f).get.map(_.toDouble)
      }

    def fitDoubleZ(features: Array[Array[Double]], labels: Array[Double], epochs: Int, lr: Double = 0.001): Task[FitResult] =
      ZIO.attemptBlocking {
        val f = features.map(_.map(_.toFloat))
        val l = labels.map(_.toFloat)
        model.fit(f, l, epochs, lr.toFloat).get
      }

    /** Stream predictions — each chunk is immediately predicted. */
    def predictFlow: ZPipeline[Any, Throwable, Array[Array[Float]], Array[Float]] =
      ZPipeline.mapZIO(features => predictZ(features))

    /** Stream training — each chunk triggers a fit() call (online SGD). */
    def fitFlow(epochs: Int = 1, lr: Float = 0.001f): ZPipeline[Any, Throwable, (Array[Array[Float]], Array[Float]), FitResult] =
      ZPipeline.mapZIO((feats, labels) => fitZ(feats, labels, epochs, lr))

  def create(arch: zio.nn.ModelDef): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(ZModel.create(arch))
    )(m => ZIO.attemptBlocking(m.close()).orDie)

  def load(file: File): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(ZModel.load(file).get)
    )(m => ZIO.attemptBlocking(m.close()).orDie)
