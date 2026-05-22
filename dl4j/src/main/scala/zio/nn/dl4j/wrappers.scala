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
  def fit(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[Unit] =
    Try {
      val ds = new DataSet(Nd4j.create(features), Nd4j.create(labels.map(Array(_))))
      for _ <- 1 to epochs do underlying.fit(ds)
    }

  /** ESCAPE HATCH: raw DL4J prediction with INDArray. */
  def predictRaw(input: INDArray): Try[INDArray] =
    Try(underlying.output(input))

  def score(dataset: DataSet): Try[Double] = Try(underlying.score(dataset))
  def save(file: File): Try[Unit] = Try(ModelSerializer.writeModel(underlying, file, false))
  def close(): Unit = underlying.close()

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
