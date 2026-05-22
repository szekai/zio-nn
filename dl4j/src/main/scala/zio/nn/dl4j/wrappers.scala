package zio.nn.dl4j

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import scala.util.Try

// ═══════════════════════════════════════════════
//  ZModel — wraps MultiLayerNetwork
// ═══════════════════════════════════════════════
class ZModel(val underlying: MultiLayerNetwork):
  def predict(input: INDArray): Try[INDArray] = Try(underlying.output(input))
  def fit(dataset: DataSet): Try[Unit] = Try(underlying.fit(dataset))
  def score(dataset: DataSet): Try[Double] = Try(underlying.score(dataset))
  def save(file: File): Try[Unit] = Try(ModelSerializer.writeModel(underlying, file, false))
  def close(): Unit = underlying.close()

object ZModel:
  def load(file: File): Try[ZModel] = Try(ZModel(ModelSerializer.restoreMultiLayerNetwork(file)))
  def wrap(net: MultiLayerNetwork): ZModel = ZModel(net)

// ═══════════════════════════════════════════════
//  ZGraphModel — wraps ComputationGraph
// ═══════════════════════════════════════════════
class ZGraphModel(val underlying: ComputationGraph):
  def predict(inputs: Array[INDArray]): Try[Array[INDArray]] =
    Try(underlying.output(false, inputs*))
  def fit(dataset: DataSet): Try[Unit] = Try(underlying.fit(dataset))
  def score(dataset: DataSet): Try[Double] = Try(underlying.score(dataset))
  def close(): Unit = underlying.close()

object ZGraphModel:
  def wrap(graph: ComputationGraph): ZGraphModel = ZGraphModel(graph)

// ═══════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════
def toDataSet(features: Array[Array[Float]], labels: Array[Float]): Try[DataSet] = Try {
  val f = Nd4j.create(features)
  val l = Nd4j.create(labels.map(Array(_)))
  new DataSet(f, l)
}
