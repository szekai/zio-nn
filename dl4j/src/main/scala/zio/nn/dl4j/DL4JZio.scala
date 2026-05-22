package zio.nn.dl4j

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import scala.util.Try

/** ZIO-friendly wrapper over DL4J Java API.
  *
  * All methods return [[scala.util.Try]]. Resources released with `.close()`.
  *
  * == Escape Hatch ==
  * Every wrapper exposes `.underlying` for direct DL4J API access:
  * {{{
  *   val model: ZModel = ...
  *   val rawNet: MultiLayerNetwork = model.underlying
  *   rawNet.getLayers.foreach(println)
  *   // Spark distributed training, custom listeners, etc.
  * }}}
  */
object DL4JZio:

  // ═══════════════════════════════════════════════════════
  //  ZModel — wraps MultiLayerNetwork
  // ═══════════════════════════════════════════════════════

  final class ZModel(val underlying: MultiLayerNetwork):
    def predict(input: INDArray): Try[INDArray] =
      Try(underlying.output(input))

    def fit(dataset: DataSet): Try[Unit] =
      Try(underlying.fit(dataset))

    def score(dataset: DataSet): Try[Double] =
      Try(underlying.score(dataset))

    def save(file: File): Try[Unit] =
      Try(ModelSerializer.writeModel(underlying, file, false))

    def close(): Unit = underlying.close()

  object ZModel:
    def load(file: File): Try[ZModel] = Try {
      new ZModel(ModelSerializer.restoreMultiLayerNetwork(file))
    }

    def wrap(net: MultiLayerNetwork): ZModel = new ZModel(net)

  // ═══════════════════════════════════════════════════════
  //  ZGraphModel — wraps ComputationGraph
  // ═══════════════════════════════════════════════════════

  final class ZGraphModel(val underlying: ComputationGraph):
    def predict(inputs: Array[INDArray]): Try[Array[INDArray]] =
      Try(underlying.output(false, inputs*))

    def fit(dataset: DataSet): Try[Unit] =
      Try(underlying.fit(dataset))

    def score(dataset: DataSet): Try[Double] =
      Try(underlying.score(dataset))

    def close(): Unit = underlying.close()

  object ZGraphModel:
    def wrap(graph: ComputationGraph): ZGraphModel = new ZGraphModel(graph)

  // ═══════════════════════════════════════════════════════
  //  Data helpers
  // ═══════════════════════════════════════════════════════

  def toDataSet(features: Array[Array[Float]], labels: Array[Float]): Try[DataSet] = Try {
    val f = Nd4j.create(features)
    val l = Nd4j.create(labels.map(Array(_)))
    new DataSet(f, l)
  }
