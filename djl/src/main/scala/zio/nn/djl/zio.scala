package zio.nn.djl

import zio.*
import java.nio.file.Path

/** ZIO-native API — Task-based with Scope resource management.
  * Import to use ZIO instead of Try:
  * {{{
  *   import zio.nn.djl.zioApi.*
  *   create(arch, "model").flatMap(_.predictZ(features))
  * }}}
  */
object zioApi:

  extension (model: ZModel)
    def predictZ(features: Array[Array[Float]]): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predict(features).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Task[ai.djl.training.TrainingResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def predictDoubleZ(features: Array[Array[Double]]): Task[Array[Double]] =
      ZIO.attemptBlocking {
        val f = features.map(_.map(_.toFloat))
        model.predict(f).get.map(_.toDouble)
      }

    def fitDoubleZ(features: Array[Array[Double]], labels: Array[Double], epochs: Int, lr: Double = 0.001): Task[ai.djl.training.TrainingResult] =
      ZIO.attemptBlocking {
        val f = features.map(_.map(_.toFloat))
        val l = labels.map(_.toFloat)
        model.fit(f, l, epochs, lr.toFloat).get
      }

  def create(arch: zio.nn.ModelDef, name: String = "model", engine: String = "PyTorch"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(ZModel.create(arch, name, engine).get)
    )(m => ZIO.attemptBlocking(m.close()).orDie)

  def load(path: Path, name: String = "model", engine: String = "PyTorch"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(ZModel.load(path, name, engine).get)
    )(m => ZIO.attemptBlocking(m.close()).orDie)
