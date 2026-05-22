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
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.Tracker
import ai.djl.util.Progress
import java.nio.file.Path
import scala.util.Try

// ═══════════════════════════════════════════════
//  ZModel — unified API across backends
// ═══════════════════════════════════════════════
class ZModel(val underlying: Model, ndm: NDManager):

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
  def fit(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[TrainingResult] =
    Try {
      val adam = Adam.builder().optLearningRateTracker(Tracker.fixed(lr)).build()
      val config = new DefaultTrainingConfig(Loss.l2Loss())
      config.optOptimizer(adam)
      config.optInitializer(new XavierInitializer(), "weight")

      val trainer = underlying.newTrainer(config)
      try
        trainer.initialize(new Shape(1, features.head.length.toLong))
        for _ <- 1 to epochs do
          val dataArr = ndm.create(features)
          val labelArr = ndm.create(labels.map(Array(_)))
          val batch = new ai.djl.training.dataset.Batch(
            ndm.newSubManager(),
            new NDList(dataArr),
            new NDList(labelArr),
            features.length, null, null, features.length.toLong, 0L, java.util.Collections.emptyList[Any]()
          )
          ai.djl.training.EasyTrain.trainBatch(trainer, batch)
          batch.close()
        trainer.getTrainingResult
      finally trainer.close()
    }

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
    // Initialize parameters via a dummy forward pass
    val inputSize = arch match
      case ModelDef.Sequential(s) => s.inputSize
      case _ => 1
    val config = new DefaultTrainingConfig(Loss.l2Loss())
    val trainer = m.newTrainer(config)
    try trainer.initialize(new Shape(1, inputSize)) finally trainer.close()
    ZModel(m, ndm)
  }

  def load(path: Path, name: String = "model", engine: String = "PyTorch"): Try[ZModel] = Try {
    val ndm = NDManager.newBaseManager()
    val m = Model.newInstance(name, ndm.getDevice(), engine)
    m.load(path, name)
    ZModel(m, ndm)
  }

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
