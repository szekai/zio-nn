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

// ═══════════════════════════════════════════════
//  ZModel — wraps ai.djl.Model
// ═══════════════════════════════════════════════
class ZModel(val underlying: Model, ndm: NDManager):
  def predictor(translator: Translator[NDList, NDList] = new NoopTranslator()): Try[ZPredictor] =
    Try(ZPredictor(new Predictor(underlying, translator, Device.cpu(), false), ndm.newSubManager()))

  def trainer(config: DefaultTrainingConfig): Try[ZTrainer] =
    Try(ZTrainer(underlying.newTrainer(config), ndm.newSubManager()))

  def save(path: Path, name: String): Try[Unit] =
    Try(underlying.save(path, name))

  def close(): Unit = { underlying.close(); ndm.close() }

object ZModel:
  def load(path: Path, name: String, engine: String = "PyTorch"): Try[ZModel] = Try {
    val ndm = NDManager.newBaseManager()
    val m = Model.newInstance(name, ndm.getDevice(), engine)
    m.load(path, name)
    ZModel(m, ndm)
  }

  def create(block: Block, name: String = "model", engine: String = "PyTorch"): Try[ZModel] = Try {
    val ndm = NDManager.newBaseManager()
    val m = Model.newInstance(name, ndm.getDevice(), engine)
    m.setBlock(block)
    ZModel(m, ndm)
  }

// ═══════════════════════════════════════════════
//  ZPredictor — wraps DJL Predictor
// ═══════════════════════════════════════════════
case class ZPredictor(underlying: Predictor[NDList, NDList], ndm: NDManager):
  def predict(input: NDList): Try[NDList] = Try(underlying.predict(input))
  def close(): Unit = { underlying.close(); ndm.close() }

// ═══════════════════════════════════════════════
//  ZTrainer — wraps DJL Trainer
// ═══════════════════════════════════════════════
case class ZTrainer(underlying: Trainer, ndm: NDManager):
  def fit(epochs: Int, trainSet: Dataset, testSet: Dataset): Try[TrainingResult] = Try {
    EasyTrain.fit(underlying, epochs, trainSet, testSet)
    underlying.getTrainingResult
  }
  def initialize(shape: Shape): Try[Unit] = Try(underlying.initialize(shape))
  def close(): Unit = { underlying.close(); ndm.close() }

// ═══════════════════════════════════════════════
//  Factory functions
// ═══════════════════════════════════════════════
def mseLoss: Loss = Loss.l2Loss()

def defaultConfig(loss: Loss = Loss.l2Loss(), lr: Float = 0.001f): DefaultTrainingConfig =
  val adam = Adam.builder().optLearningRateTracker(Tracker.fixed(lr)).build()
  val config = new DefaultTrainingConfig(loss)
  config.optOptimizer(adam)
  config.optInitializer(new XavierInitializer(), "weight")
  config
