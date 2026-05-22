package zio.nn.djl

import ai.djl.{Model, Device}
import ai.djl.ndarray.{NDList, NDManager, NDArray}
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Block
import ai.djl.inference.Predictor
import ai.djl.translate.{Translator, NoopTranslator}
import ai.djl.training.{Trainer, DefaultTrainingConfig, EasyTrain, TrainingResult}
import ai.djl.training.dataset.Dataset
import ai.djl.training.loss.Loss
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.optimizer.Adam
import ai.djl.training.tracker.Tracker
import java.nio.file.Path
import scala.util.Try

/** ZIO-friendly wrapper over DJL Java API.
  *
  * All methods return [[scala.util.Try]] — pattern-match on `Success`/`Failure`.
  * Resources must be released with `.close()`.
  *
  * == Escape Hatch ==
  * Every wrapper exposes `.underlying` for direct access to the raw DJL object:
  * {{{
  *   val model: ZModel = ...
  *   val rawDJLModel: ai.djl.Model = model.underlying
  *   // Now use any DJL API directly
  *   rawDJLModel.getBlock().getChildren().forEach(...)
  * }}}
  * This is the intended path for multi-GPU training, fine-tuning, custom layers,
  * and any feature not covered by this thin wrapper.
  */
object DJLZio:

  // ═══════════════════════════════════════════════════════
  //  ZModel
  // ═══════════════════════════════════════════════════════

  final class ZModel(val underlying: Model, ndm: NDManager):
    def predictor(translator: Translator[NDList, NDList] = new NoopTranslator()): Try[ZPredictor] =
      Try(new ZPredictor(new Predictor(underlying, translator, Device.cpu(), false), ndm.newSubManager()))

    def trainer(config: DefaultTrainingConfig): Try[ZTrainer] =
      Try(new ZTrainer(underlying.newTrainer(config), ndm.newSubManager()))

    def save(path: Path, name: String): Try[Unit] =
      Try(underlying.save(path, name))

    def close(): Unit = { underlying.close(); ndm.close() }

  object ZModel:
    def load(path: Path, name: String, engine: String = "PyTorch"): Try[ZModel] = Try {
      val ndm = NDManager.newBaseManager()
      val m = Model.newInstance(name, ndm.getDevice(), engine)
      m.load(path, name)
      new ZModel(m, ndm)
    }

    def create(block: Block, name: String = "model", engine: String = "PyTorch"): Try[ZModel] = Try {
      val ndm = NDManager.newBaseManager()
      val m = Model.newInstance(name, ndm.getDevice(), engine)
      m.setBlock(block)
      new ZModel(m, ndm)
    }

  // ═══════════════════════════════════════════════════════
  //  ZPredictor
  // ═══════════════════════════════════════════════════════

  final class ZPredictor(val underlying: Predictor[NDList, NDList], ndm: NDManager):
    def predict(input: NDList): Try[NDList] =
      Try(underlying.predict(input))

    def close(): Unit = { underlying.close(); ndm.close() }

  // ═══════════════════════════════════════════════════════
  //  ZTrainer
  // ═══════════════════════════════════════════════════════

  final class ZTrainer(val underlying: Trainer, ndm: NDManager):
    def fit(epochs: Int, trainSet: Dataset, testSet: Dataset): Try[TrainingResult] = Try {
      EasyTrain.fit(underlying, epochs, trainSet, testSet)
      underlying.getTrainingResult
    }

    def initialize(shape: Shape): Try[Unit] =
      Try(underlying.initialize(shape))

    def close(): Unit = { underlying.close(); ndm.close() }

  // ═══════════════════════════════════════════════════════
  //  Factory helpers
  // ═══════════════════════════════════════════════════════

  def mseLoss: Loss = Loss.l2Loss()

  def defaultConfig(loss: Loss = Loss.l2Loss(), lr: Float = 0.001f): DefaultTrainingConfig =
    val adam = Adam.builder().optLearningRateTracker(Tracker.fixed(lr)).build()
    val config = new DefaultTrainingConfig(loss)
    config.optOptimizer(adam)
    config.optInitializer(new XavierInitializer(), "weight")
    config
