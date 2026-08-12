package zio.nn.xgboost

import ml.dmlc.xgboost4j.java.{Booster, DMatrix, XGBoost as XGBoostJ}
import zio.nn.{FitResult, LossFn, ModelDef, OptimizerDef}
import java.io.File
import java.nio.file.Path
import scala.util.{Failure, Success, Try}

/** Gradient-boosted tree model wrapping an xgboost4j `Booster`.
  *
  * Same duck-typed contract as the other zio-nn backends:
  *   - `predict(features)`      — one regression output per row
  *   - `predictDirect`          — RNN-shaped input flattened to a single sample
  *   - `fit` / `fitArray3D`     — train trees (objective `reg:squarederror`)
  *   - `save` / `load`          — `model.json` inside the given directory
  */
final class ZModel private (
    private var booster: Option[Booster],
    val inputSize: Int,
    val outputSize: Int
):
  private val defaultObjective = "reg:squarederror"

  def predict(features: Array[Array[Float]]): Try[Array[Float]] = Try {
    requireBooster
    val dmat = toDMatrix(features, labels = None)
    try booster.get.predict(dmat, false, 0).map(_(0))
    finally dmat.dispose()
  }

  /** RNN-shaped input (timeSteps × featCount) flattened to one tree sample;
    * the prediction is replicated across timeSteps so `predictions.head ==
    * predictions.last` (both consumers use one of them as the final read).
    */
  def predictDirect(features: Array[Array[Float]], timeSteps: Int, featCount: Int): Try[Array[Float]] =
    predict(Array(features.flatten)).map(pred => Array.fill(timeSteps)(pred.head))

  def fit(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[FitResult] =
    train(features, labels, epochs, lr)

  def fit(features: Array[Array[Float]], labels: Array[Array[Float]], epochs: Int, lr: Float): Try[FitResult] =
    train(features, labels.map(_(0)), epochs, lr)

  def fit(features: Array[Array[Float]], labels: Array[Int], epochs: Int, lr: Float): Try[FitResult] =
    train(features, labels.map(_.toFloat), epochs, lr)

  /** 3D RNN input (samples × timeSteps × features) flattened per sample. */
  def fitArray3D(features: Array[Array[Array[Float]]], labels: Array[Float], epochs: Int, batchSize: Int, lr: Float = 0.001f): Try[FitResult] =
    train(features.map(_.flatten), labels, epochs, lr)

  /** Streamed batches collected into one flat training set. */
  def fitDataset(dataset: zio.nn.DataSetLoader, epochs: Int, batchSize: Int, lr: Float = 0.001f): Try[FitResult] =
    val chunks = zio.Unsafe.unsafe { implicit u =>
      zio.Runtime.default.unsafe.run(dataset.batches.runCollect).getOrThrow()
    }
    val features = chunks.flatMap(_._1).toArray
    val labels = chunks.flatMap(_._2).toArray
    train(features, labels, epochs, lr)

  /** 3D streamed batches (each sample = flat timeSteps×features) — trees see
    * the same flat sample, so this delegates to [[fitDataset]].
    */
  def fitDataset3D(dataset: zio.nn.DataSetLoader, epochs: Int, batchSize: Int, timeSteps: Int, featuresPerBar: Int, lr: Float = 0.001f): Try[FitResult] =
    fitDataset(dataset, epochs, batchSize, lr)

  def evaluate(features: Array[Array[Float]], labels: Array[Float], metrics: List[zio.nn.EvalMetric]): Try[Map[String, Double]] =
    predict(features).map { preds =>
      val base = Map("mse" -> LossFn.MSE.compute(preds.map(_.toDouble), labels.map(_.toDouble)))
      metrics.foldLeft(base) { (acc, m) => acc + (m.toString -> m.compute(preds.map(_.toDouble), labels.map(_.toDouble))) }
    }

  def evaluateDataset(dataset: zio.nn.DataSetLoader, batchSize: Int, metrics: List[zio.nn.EvalMetric]): Try[Map[String, Double]] =
    val chunks = zio.Unsafe.unsafe { implicit u =>
      zio.Runtime.default.unsafe.run(dataset.batches.runCollect).getOrThrow()
    }
    evaluate(chunks.flatMap(_._1).toArray, chunks.flatMap(_._2).toArray, metrics)

  def save(path: Path): Try[Unit] = Try {
    requireBooster
    val dir = path.toFile
    if !dir.exists() then dir.mkdirs()
    booster.get.saveModel(new File(dir, "model.json").getAbsolutePath)
  }

  def close(): Unit =
    booster.foreach(_.dispose())
    booster = None

  def summary: String = s"XGBoost(inputSize=$inputSize, outputSize=$outputSize, booster=${booster.isDefined})"

  // ── internals ──

  private def requireBooster: Unit =
    if booster.isEmpty then throw new IllegalStateException("Model is not trained — call fit() first")

  private def toDMatrix(features: Array[Array[Float]], labels: Option[Array[Float]]): DMatrix =
    val nrow = features.length
    val ncol = if nrow == 0 then 0 else features(0).length
    val dmat = new DMatrix(features.flatten, nrow, ncol, Float.NaN)
    labels.foreach(dmat.setLabel)
    dmat

  private def train(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float): Try[FitResult] =
    if features.isEmpty || features.length != labels.length then
      Failure(new IllegalArgumentException("features and labels must be non-empty and same length"))
    else Try {
      val dmat = toDMatrix(features, Some(labels))
      try
        val params = new java.util.HashMap[String, AnyRef]()
        params.put("eta", lr.toDouble.asInstanceOf[AnyRef])
        params.put("objective", defaultObjective)
        params.put("seed", 42L.asInstanceOf[AnyRef])
        // XGBoost.train runs `epochs` boosting rounds internally (the only
        // public training entry point — createBooster/update are private[java]).
        // `watches` must be non-null (empty map), obj/eval may be null.
        val b = XGBoostJ.train(dmat, params, epochs, new java.util.HashMap[String, DMatrix](), null, null)
        val preds = b.predict(dmat, false, 0).map(_(0))
        val loss = LossFn.MSE.compute(preds.map(_.toDouble), labels.map(_.toDouble))
        booster = Some(b)
        FitResult(loss = loss, epochs = epochs, lossHistory = List(loss))
      finally dmat.dispose()
    }

object ZModel:

  def create(arch: ModelDef, name: String = "model", engine: String = "XGBoost", device: String = "cpu"): Try[ZModel] =
    Backend.compile(arch).map(cfg => new ZModel(None, cfg.inputSize, cfg.outputSize))

  def load(path: Path, name: String = "model", engine: String = "XGBoost", device: String = "cpu"): Try[ZModel] =
    Try {
      val modelFile = new File(path.toFile, "model.json")
      if !modelFile.exists() then
        throw new IllegalArgumentException(s"no trained XGBoost model at $modelFile (run fit() and save() first)")
      new ZModel(Some(XGBoostJ.loadModel(modelFile.getAbsolutePath)), inputSize = -1, outputSize = 1)
    }
