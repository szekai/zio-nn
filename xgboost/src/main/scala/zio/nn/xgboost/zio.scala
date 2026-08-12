package zio.nn.xgboost

import zio.*
import zio.nn.{DataSetLoader, FitResult, ModelDef}
import java.nio.file.Path

/** ZIO-native surface mirroring `zio.nn.djl.zioApi` — `create`/`load` are
  * scope-managed (auto-closed); fit/predict are blocking Try calls wrapped in
  * `ZIO.attemptBlocking`.
  */
object zioApi:

  extension (model: ZModel)
    def predictZ(features: Array[Array[Float]]): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predict(features).get)

    def predictDirectZ(features: Array[Array[Float]], timeSteps: Int, featCount: Int): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predictDirect(features, timeSteps, featCount).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Array[Float]], epochs: Int, lr: Float): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Int], epochs: Int, lr: Float): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def fitArray3DZ(features: Array[Array[Array[Float]]], labels: Array[Float], epochs: Int, batchSize: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fitArray3D(features, labels, epochs, batchSize, lr).get)

    def fitDatasetZ(dataset: DataSetLoader, epochs: Int, batchSize: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fitDataset(dataset, epochs, batchSize, lr).get)

    def fitDataset3DZ(dataset: DataSetLoader, epochs: Int, batchSize: Int, timeSteps: Int, featuresPerBar: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fitDataset3D(dataset, epochs, batchSize, timeSteps, featuresPerBar, lr).get)

    def evaluateZ(features: Array[Array[Float]], labels: Array[Float], metrics: List[zio.nn.EvalMetric]): Task[Map[String, Double]] =
      ZIO.attemptBlocking(model.evaluate(features, labels, metrics).get)

    /** Train in `saveEvery`-epoch chunks, saving `model.json` at
      * `$checkpointPath-epoch$n` after each chunk (mirrors the DJL backend).
      */
    def fitWithCheckpoints(
        features: Array[Array[Float]],
        labels: Array[Float],
        epochs: Int,
        saveEvery: Int,
        checkpointPath: String,
        lr: Float = 0.001f
    ): ZIO[Any, Throwable, FitResult] =
      val chunkSizes =
        var remaining = math.max(0, epochs)
        val sizes = List.newBuilder[Int]
        while remaining > 0 do
          sizes += math.min(saveEvery, remaining)
          remaining -= saveEvery
        sizes.result()
      var savedEpoch = 0
      ZIO.foldLeft(chunkSizes)(FitResult(0.0, 0, Nil)) { (_, chunkEpochs) =>
        savedEpoch += chunkEpochs
        model.fitZ(features, labels, chunkEpochs, lr) <* ZIO.attemptBlocking(
          model.save(Path.of(s"$checkpointPath-epoch$savedEpoch")).get
        )
      }

  def create(arch: ModelDef, name: String = "model", engine: String = "XGBoost"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attempt(ZModel.create(arch, name, engine).get)
    )(m => ZIO.attempt(m.close()).ignore)

  def load(path: Path, name: String = "model", engine: String = "XGBoost"): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(
      ZIO.attempt(ZModel.load(path, name, engine).get)
    )(m => ZIO.attempt(m.close()).ignore)
