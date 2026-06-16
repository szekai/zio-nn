package zio.nn.djl

import zio.nn.*
import ai.djl.{Model, Device}
import ai.djl.ndarray.{NDList, NDManager, NDArray}
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Block
import ai.djl.inference.Predictor
import ai.djl.translate.{Translator, NoopTranslator}
import ai.djl.training.{Trainer, DefaultTrainingConfig, EasyTrain, TrainingResult}
import ai.djl.training.dataset.{Dataset, RandomAccessDataset, Record}
import ai.djl.training.loss.Loss
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.optimizer.{Adam, Sgd, RmsProp}
import ai.djl.training.tracker.Tracker
import ai.djl.util.Progress
import java.nio.file.Path
import scala.util.Try

// ═══════════════════════════════════════════════
//  ZModel — unified API across backends
// ═══════════════════════════════════════════════
class ZModel(val underlying: Model, ndm: NDManager, lossFn: LossFn, optimizerDef: OptimizerDef = OptimizerDef.Adam()):

  /** UNIFIED: predict from float arrays. Works identically on both backends. */
  def predict(features: Array[Array[Float]]): Try[Array[Float]] =
    val sub = ndm.newSubManager()
    try
      val input = new NDList(sub.create(features))
      val pred  = Predictor(underlying, new NoopTranslator(), Device.cpu(), false)
      try
        val result = pred.predict(input)
        val arr = new Array[Float](result.head().size().toInt)
        val src = result.head().toFloatArray
        System.arraycopy(src, 0, arr, 0, arr.length)
        Try(arr)
      finally pred.close()
    finally sub.close()

  /** UNIFIED: train from float arrays. Works identically on both backends. */
  def fit(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = LR_UNSPECIFIED): Try[FitResult] =
    Try {
      val opt = selectOptimizer(lr)
      val config = new DefaultTrainingConfig(selectLoss(lossFn))
      config.optOptimizer(opt)
      config.optInitializer(new XavierInitializer(), "weight")
      val trainer = underlying.newTrainer(config)
      try
        trainer.initialize(new Shape(1, features.head.length.toLong))
        val lossHistory = scala.collection.mutable.ListBuffer[Double]()
        for _ <- 1 to epochs do
          val dataArr = ndm.create(features); val labelArr = ndm.create(labels.map(Array(_)))
          val batch = new ai.djl.training.dataset.Batch(ndm.newSubManager(), new NDList(dataArr), new NDList(labelArr),
            features.length, null, null, features.length.toLong, 0L, java.util.Collections.emptyList[Any]())
          ai.djl.training.EasyTrain.trainBatch(trainer, batch); batch.close()
          lossHistory += trainer.getTrainingResult.getTrainLoss.toDouble
        FitResult(lossHistory.lastOption.getOrElse(Double.NaN), epochs, lossHistory.toList)
      finally trainer.close()
    }

  def fit(features: Array[Array[Float]], labels: Array[Array[Float]], epochs: Int, lr: Float): Try[FitResult] =
    Try {
      val opt = selectOptimizer(lr)
      val config = new DefaultTrainingConfig(selectLoss(lossFn))
      config.optOptimizer(opt)
      config.optInitializer(new XavierInitializer(), "weight")
      val trainer = underlying.newTrainer(config)
      try
        trainer.initialize(new Shape(1, features.head.length.toLong))
        val lossHistory = scala.collection.mutable.ListBuffer[Double]()
        for _ <- 1 to epochs do
          val dataArr = ndm.create(features); val labelArr = ndm.create(labels)
          val batch = new ai.djl.training.dataset.Batch(ndm.newSubManager(), new NDList(dataArr), new NDList(labelArr),
            features.length, null, null, features.length.toLong, 0L, java.util.Collections.emptyList[Any]())
          ai.djl.training.EasyTrain.trainBatch(trainer, batch); batch.close()
          lossHistory += trainer.getTrainingResult.getTrainLoss.toDouble
        FitResult(lossHistory.lastOption.getOrElse(Double.NaN), epochs, lossHistory.toList)
      finally trainer.close()
    }

  def fit(features: Array[Array[Float]], labels: Array[Int], epochs: Int, lr: Float): Try[FitResult] =
    Try {
      val numClasses = labels.max + 1
      val oneHot = labels.map { label =>
        val row = new Array[Float](numClasses); row(label) = 1.0f; row
      }
      val opt = selectOptimizer(lr)
      val config = new DefaultTrainingConfig(selectLoss(lossFn))
      config.optOptimizer(opt)
      config.optInitializer(new XavierInitializer(), "weight")
      val trainer = underlying.newTrainer(config)
      try
        trainer.initialize(new Shape(1, features.head.length.toLong))
        val lossHistory = scala.collection.mutable.ListBuffer[Double]()
        for _ <- 1 to epochs do
          val dataArr = ndm.create(features); val labelArr = ndm.create(oneHot)
          val batch = new ai.djl.training.dataset.Batch(ndm.newSubManager(), new NDList(dataArr), new NDList(labelArr),
            features.length, null, null, features.length.toLong, 0L, java.util.Collections.emptyList[Any]())
          ai.djl.training.EasyTrain.trainBatch(trainer, batch); batch.close()
          lossHistory += trainer.getTrainingResult.getTrainLoss.toDouble
        FitResult(lossHistory.lastOption.getOrElse(Double.NaN), epochs, lossHistory.toList)
      finally trainer.close()
    }

  private def selectLoss(lossFn: LossFn): Loss = lossFn match
    case LossFn.MSE                     => Loss.l2Loss()
    case LossFn.MAE                     => Loss.l1Loss()
    case LossFn.BinaryCrossEntropy(_)   => Loss.sigmoidBinaryCrossEntropyLoss()
    case LossFn.CategoricalCrossEntropy(_) => Loss.softmaxCrossEntropyLoss()
    case LossFn.Huber(_)                => Loss.l2Loss() // DJL has no native Huber

  private val LR_UNSPECIFIED = -1.0f

  private def selectOptimizer(lr: Float): Optimizer =
    val effectiveLr =
      if lr == LR_UNSPECIFIED then optimizerDef match
        case OptimizerDef.Adam(r)   => r.toFloat
        case OptimizerDef.SGD(r)    => r.toFloat
        case OptimizerDef.RMSprop(r) => r.toFloat
      else lr
    optimizerDef match
      case OptimizerDef.Adam(_)    => Adam.builder().optLearningRateTracker(Tracker.fixed(effectiveLr)).build()
      case OptimizerDef.SGD(_)     => Sgd.builder().setLearningRateTracker(Tracker.fixed(effectiveLr)).build()
      case OptimizerDef.RMSprop(_) => RmsProp.builder().optLearningRateTracker(Tracker.fixed(effectiveLr)).build()

  /** ESCAPE HATCH: raw DJL prediction with NDList. */
  def predictRaw(input: NDList): Try[NDList] = Try {
    val pred = Predictor(underlying, new NoopTranslator(), Device.cpu(), false)
    try pred.predict(input) finally pred.close()
  }

  /** ESCAPE HATCH: create a DJL Predictor for multi-step inference. */
  def predictorRaw(): ZPredictor =
    ZPredictor(Predictor(underlying, new NoopTranslator(), Device.cpu(), false), ndm.newSubManager())

  /** ESCAPE HATCH: create a DJL Trainer for custom training loops. */
  def trainerRaw(config: DefaultTrainingConfig): ZTrainer =
    ZTrainer(underlying.newTrainer(config), ndm.newSubManager())

  def save(path: Path): Try[Unit] = Try(underlying.save(path, "model"))
  def close(): Unit = { underlying.close(); ndm.close() }

  /** Predict with integer token indices (when first layer is Embedding). */
  def predictInt(tokens: Array[Array[Int]]): Try[Array[Float]] =
    val sub = ndm.newSubManager()
    try
      val flatTokens = tokens.flatten.map(_.toLong)
      val input = new NDList(sub.create(flatTokens, new Shape(tokens.length.toLong, tokens.head.length.toLong)))
      val pred = Predictor(underlying, new NoopTranslator(), Device.cpu(), false)
      try
        val result = pred.predict(input)
        val head = result.head()
        val arr = new Array[Float](head.size().toInt)
        System.arraycopy(head.toFloatArray, 0, arr, 0, arr.length)
        Try(arr)
      finally pred.close()
    finally sub.close()

  /** Train with integer token indices (when first layer is Embedding). */
  def fitInt(tokens: Array[Array[Int]], labels: Array[Float], epochs: Int, lr: Float = LR_UNSPECIFIED): Try[FitResult] =
    Try {
      val opt = selectOptimizer(lr)
      val config = new DefaultTrainingConfig(selectLoss(lossFn))
      config.optOptimizer(opt)
      config.optInitializer(new XavierInitializer(), "weight")
      val trainer = underlying.newTrainer(config)
      try
        val flatTokens = tokens.flatten.map(_.toLong)
        trainer.initialize(new Shape(tokens.length.toLong, tokens.head.length.toLong))
        val lossHistory = scala.collection.mutable.ListBuffer[Double]()
        for _ <- 1 to epochs do
          val dataArr = ndm.create(flatTokens, new Shape(tokens.length.toLong, tokens.head.length.toLong))
          val labelArr = ndm.create(labels.map(_.toFloat))
          val batch = new ai.djl.training.dataset.Batch(ndm.newSubManager(), new NDList(dataArr), new NDList(labelArr),
            tokens.length, null, null, tokens.length.toLong, 0L, java.util.Collections.emptyList[Any]())
          ai.djl.training.EasyTrain.trainBatch(trainer, batch); batch.close()
          lossHistory += trainer.getTrainingResult.getTrainLoss.toDouble
        FitResult(lossHistory.lastOption.getOrElse(Double.NaN), epochs, lossHistory.toList)
      finally trainer.close()
    }

  /** Evaluate model on test data using the given metrics.
    * Returns a map of metric name → computed value.
    * Uses predict() internally, then applies each metric.
    */
  def evaluate(
    features: Array[Array[Float]],
    labels: Array[Float],
    metrics: List[zio.nn.EvalMetric]
  ): Try[Map[String, Double]] =
    predict(features).flatMap { predictions =>
      Try {
        val predDouble = predictions.map(_.toDouble)
        val actualDouble = labels.map(_.toDouble)
        // For classification: if n_out > 1, reshape predictions per sample
        val nSamples = features.length
        val nOut = predictions.length / nSamples
        if nOut == 1 then
          metrics.map(m => m.name -> m.compute(predDouble, actualDouble)).toMap
        else
          // Multi-class: reshape to (samples, classes), take argmax per sample
          val pred2D = predDouble.grouped(nOut).toArray
          val act2D  = actualDouble.grouped(nOut).toArray
          val predClass = pred2D.map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
          val actClass  = act2D.map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
          metrics.map(m => m.name -> m.compute(predClass, actClass)).toMap
      }
    }

  def predictAndStore(
    features: Array[Array[Float]],
    store: zio.nn.VectorStore,
    ids: Array[String]
  ): Try[Array[Float]] =
    predict(features).flatMap { predictions =>
      val nSamples = features.length
      val nOut = predictions.length / nSamples
      val records = if nOut == 1 then
        ids.zip(predictions).map { case (id, value) =>
          zio.nn.VectorRecord(id, Array(value))
        }
      else
        val pred2D = predictions.grouped(nOut).toArray
        ids.zip(pred2D).map { case (id, vec) =>
          zio.nn.VectorRecord(id, vec)
        }
      store.storeBatch(records).map(_ => predictions)
    }

  def summary: String =
    val sb = new StringBuilder(s"ZModel[${underlying.getName}]\n")
    val children = underlying.getBlock.getChildren
    val it = children.iterator()
    var idx = 0
    while it.hasNext do
      val child = it.next().getValue
      sb.append(f"  $idx%3d  ${child.getClass.getSimpleName}\n")
      idx += 1
    sb.toString

object ZModel:

  /** UNIFIED: create a model from architecture. Compiles + instantiates internally. */
  def create(arch: ModelDef, name: String = "model", engine: String = "PyTorch"): Try[ZModel] = Try {
    val block = Backend.compile(arch)
    val ndm = NDManager.newBaseManager()
    val m = Model.newInstance(name, ndm.getDevice(), engine)
    m.setBlock(block)
    val lossFn = extractLoss(arch)
    val optimizer = arch match
      case ModelDef.Sequential(s) => s.optimizer
      case ModelDef.Functional(f) => f.optimizer
    val inputSize = arch match
      case ModelDef.Sequential(s) => s.inputSize
      case _ => 1
    val config = new DefaultTrainingConfig(Loss.l2Loss())
    val trainer = m.newTrainer(config)
    try trainer.initialize(new Shape(1, inputSize)) finally trainer.close()
    ZModel(m, ndm, lossFn, optimizer)
  }

  private def extractLoss(arch: ModelDef): LossFn = arch match
    case ModelDef.Sequential(s) =>
      s.layers.collectFirst { case AnyLayer.Standard(LayerDef.Output(_, _, loss, _)) => loss }.getOrElse(LossFn.MSE)
    case ModelDef.Functional(f) =>
      f.layers.collectFirst { case (_, LayerDef.Output(_, _, loss, _)) => loss }.getOrElse(LossFn.MSE)

  def load(path: Path, name: String = "model", engine: String = "PyTorch"): Try[ZModel] = Try {
    val ndm = NDManager.newBaseManager()
    val m = Model.newInstance(name, ndm.getDevice(), engine)
    m.load(path, name)
    ZModel(m, ndm, LossFn.MSE)
  }

  /** Load an ONNX model. Convenience method that defaults engine to "OnnxRuntime". */
  def loadOnnx(path: Path, name: String = "model"): Try[ZModel] =
    load(path, name, engine = "OnnxRuntime")

// ═══════════════════════════════════════════════
//  ZPredictor / ZTrainer — escape hatch only
// ═══════════════════════════════════════════════
case class ZPredictor(underlying: Predictor[NDList, NDList], ndm: NDManager):
  def predict(input: NDList): Try[NDList] = Try(underlying.predict(input))
  def close(): Unit = { underlying.close(); ndm.close() }

case class ZTrainer(underlying: Trainer, ndm: NDManager):
  def fit(epochs: Int, trainSet: Dataset, testSet: Dataset): Try[TrainingResult] = Try {
    EasyTrain.fit(underlying, epochs, trainSet, testSet)
    underlying.getTrainingResult
  }
  def initialize(shape: Shape): Try[Unit] = Try(underlying.initialize(shape))
  def close(): Unit = { underlying.close(); ndm.close() }
