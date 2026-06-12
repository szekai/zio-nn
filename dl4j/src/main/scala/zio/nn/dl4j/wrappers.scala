package zio.nn.dl4j

import zio.nn.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import scala.util.Try

// ═══════════════════════════════════════════════
//  ZModel — unified API across backends
// ═══════════════════════════════════════════════
class ZModel(val underlying: MultiLayerNetwork):

  /** UNIFIED: predict from float arrays. Same signature as DJL backend. */
  def predict(features: Array[Array[Float]]): Try[Array[Float]] =
    Try {
      val input = Nd4j.create(features)
      val output = underlying.output(input)
      val total = output.length().toInt
      val arr = new Array[Float](total)
      for i <- arr.indices do arr(i) = output.getFloat(i.toLong)
      arr
    }

  /** UNIFIED: train from float arrays. Same signature as DJL backend. */
  def fit(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[FitResult] =
    Try {
      val ds = new DataSet(Nd4j.create(features), Nd4j.create(labels.map(Array(_))))
      val history = scala.collection.mutable.ListBuffer[Double]()
      for _ <- 1 to epochs do { underlying.fit(ds); history += underlying.score(ds) }
      FitResult(history.lastOption.getOrElse(Double.NaN), epochs, history.toList)
    }

  def fit(features: Array[Array[Float]], labels: Array[Array[Float]], epochs: Int, lr: Float): Try[FitResult] =
    Try {
      val ds = new DataSet(Nd4j.create(features), Nd4j.create(labels))
      val history = scala.collection.mutable.ListBuffer[Double]()
      for _ <- 1 to epochs do { underlying.fit(ds); history += underlying.score(ds) }
      FitResult(history.lastOption.getOrElse(Double.NaN), epochs, history.toList)
    }

  def fit(features: Array[Array[Float]], labels: Array[Int], epochs: Int, lr: Float): Try[FitResult] =
    Try {
      val numClasses = labels.max + 1
      val oneHot = labels.map { label =>
        val row = new Array[Float](numClasses)
        row(label) = 1.0f
        row
      }
      val ds = new DataSet(Nd4j.create(features), Nd4j.create(oneHot))
      val history = scala.collection.mutable.ListBuffer[Double]()
      for _ <- 1 to epochs do { underlying.fit(ds); history += underlying.score(ds) }
      FitResult(history.lastOption.getOrElse(Double.NaN), epochs, history.toList)
    }

  /** ESCAPE HATCH: raw DL4J prediction with INDArray. */
  def predictRaw(input: INDArray): Try[INDArray] =
    Try(underlying.output(input))

  def score(dataset: DataSet): Try[Double] = Try(underlying.score(dataset))
  def save(file: File): Try[Unit] = Try(ModelSerializer.writeModel(underlying, file, false))
  def close(): Unit = underlying.close()

  def summary: String =
    val sb = new StringBuilder("ZModel[MultiLayerNetwork]\n")
    val layers = underlying.getLayers
    for i <- layers.indices do
      sb.append(f"  $i%3d  ${layers(i).getClass.getSimpleName}\n")
    sb.toString

  /** Predict with integer token indices (when first layer is Embedding). */
  def predictInt(tokens: Array[Array[Int]]): Try[Array[Float]] =
    Try {
      val indArray = Nd4j.create(tokens.map(_.map(_.toFloat)))
      val output = underlying.output(indArray)
      val total = output.length().toInt
      val arr = new Array[Float](total)
      for i <- arr.indices do arr(i) = output.getFloat(i.toLong)
      arr
    }

  /** Train with integer token indices (when first layer is Embedding). */
  def fitInt(tokens: Array[Array[Int]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[FitResult] =
    Try {
      val ds = new DataSet(Nd4j.create(tokens.map(_.map(_.toFloat))), Nd4j.create(labels.map(Array(_))))
      val history = scala.collection.mutable.ListBuffer[Double]()
      for _ <- 1 to epochs do { underlying.fit(ds); history += underlying.score(ds) }
      FitResult(history.lastOption.getOrElse(Double.NaN), epochs, history.toList)
    }

object ZModel:

  /** UNIFIED: create a model from architecture. Compiles + wraps internally. */
  def create(arch: ModelDef): ZModel =
    ZModel(Backend.compile(arch))

  def load(file: File): Try[ZModel] =
    Try(ZModel(ModelSerializer.restoreMultiLayerNetwork(file)))

// ═══════════════════════════════════════════════
//  ZGraphModel — for ComputationGraph users
// ═══════════════════════════════════════════════
class ZGraphModel(val underlying: ComputationGraph):
  def predict(inputs: Array[INDArray]): Try[Array[INDArray]] =
    Try(underlying.output(false, inputs*))
  def fit(dataset: DataSet): Try[Unit] = Try(underlying.fit(dataset))
  def close(): Unit = underlying.close()

object ZGraphModel:
  def wrap(graph: ComputationGraph): ZGraphModel = ZGraphModel(graph)
