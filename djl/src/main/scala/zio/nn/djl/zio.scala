package zio.nn.djl

import zio.*
import zio.stream.*
import zio.nn.FitResult
import java.nio.file.Path

/** ZIO-native API — Task-based with Scope resource management and ZStream support.
  *
  * Usage:
  * {{{
  *   import zio.nn.djl.zioApi.*
  *
  *   // Batch prediction
  *   create(arch, "model").flatMap(_.predictZ(features))
  *
  *   // Streaming prediction — live data → predictions
  *   priceStream.via(model.predictFlow).runCollect
  *
  *   // Streaming training — online SGD from infinite stream
  *   dataStream.via(model.fitFlow(epochs = 1)).runDrain
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

    // ═══ ZStream integration ═══

    /** Stream predictions — each chunk is immediately predicted.
      * {{{
      *   val predictions: ZStream[Any, Throwable, Array[Float]] =
      *     featureStream.via(model.predictFlow)
      * }}}
      */
    def predictFlow: ZPipeline[Any, Throwable, Array[Array[Float]], Array[Float]] =
      ZPipeline.mapZIO(features => predictZ(features))

    /** Stream training — each chunk triggers a fit() call (online SGD).
      * {{{
      *   val results: ZStream[Any, Throwable, FitResult] =
      *     dataStream.via(model.fitFlow(epochs = 1))
      * }}}
      */
    def fitFlow(epochs: Int = 1, lr: Float = 0.001f): ZPipeline[Any, Throwable, (Array[Array[Float]], Array[Float]), FitResult] =
      ZPipeline.mapZIO((feats, labels) => fitZ(feats, labels, epochs, lr))

  // ═══ ZModel lifecycle ═══

  def create(arch: zio.nn.ModelDef, name: String = "model", engine: String = "PyTorch"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(ZModel.create(arch, name, engine).get)
    )(m => ZIO.attemptBlocking(m.close()).orDie)

  def load(path: Path, name: String = "model", engine: String = "PyTorch"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(ZModel.load(path, name, engine).get)
    )(m => ZIO.attemptBlocking(m.close()).orDie)
