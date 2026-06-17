package zio.nn.dl4j

import zio.nn.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.nio.file.Path
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

  /** RNN: single time-step forward pass, maintaining internal state.
    * The network retains its hidden state between calls.
    * Call rnnClearPreviousState() to reset between independent sequences.
    */
  def rnnTimeStep(input: INDArray): Try[INDArray] =
    Try(underlying.rnnTimeStep(input))

  /** RNN: reset internal state of all recurrent layers.
    * Must be called before processing a new independent sequence.
    */
  def rnnClearPreviousState(): Try[Unit] =
    Try(underlying.rnnClearPreviousState())

  /** ESCAPE HATCH: raw DL4J prediction with INDArray. */
  def predictRaw(input: INDArray): Try[INDArray] =
    Try(underlying.output(input))

  def score(dataset: DataSet): Try[Double] = Try(underlying.score(dataset))
  def save(file: File): Try[Unit] = Try(ModelSerializer.writeModel(underlying, file, false))
  def toOnnx(path: java.nio.file.Path): Try[Unit] =
    Dl4jToOnnx.saveToFile(underlying, path)
  def close(): Unit = underlying.close()

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
        val nSamples = features.length
        val nOut = predictions.length / nSamples
        if nOut == 1 then
          metrics.map(m => m.name -> m.compute(predDouble, actualDouble)).toMap
        else
          val pred2D = predDouble.grouped(nOut).toArray
          val act2D  = actualDouble.grouped(nOut).toArray
          val predClass = pred2D.map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
          val actClass  = act2D.map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
          metrics.map(m => m.name -> m.compute(predClass, actClass)).toMap
      }
    }

  /** Predict and store results in a vector store. */
  def predictAndStore(
    features: Array[Array[Float]],
    store: VectorStore,
    ids: Array[String]
  ): Try[Array[Float]] =
    predict(features).flatMap { predictions =>
      val nSamples = features.length
      val nOut = predictions.length / nSamples
      val records = if nOut == 1 then
        ids.zip(predictions).map { case (id, value) =>
          VectorRecord(id, Array(value))
        }
      else
        val pred2D = predictions.grouped(nOut).toArray
        ids.zip(pred2D).map { case (id, vec) =>
          VectorRecord(id, vec)
        }
      store.storeBatch(records).map(_ => predictions)
    }

  /** Train using a DataSetIterator (streaming from disk).
    * Iterates all batches for the specified number of epochs.
    * The iterator is reset() at the start of each epoch.
    */
  def fit(iterator: DataSetIterator, epochs: Int, lr: Float): Try[FitResult] =
    Try {
      val history = scala.collection.mutable.ListBuffer[Double]()
      for _ <- 1 to epochs do
        iterator.reset()
        while iterator.hasNext do underlying.fit(iterator.next())
        history += underlying.score()
      FitResult(history.lastOption.getOrElse(Double.NaN), epochs, history.toList)
    }

  /** Evaluate using a DataSetIterator (streaming from disk).
    * Iterates all batches, accumulates predictions, then computes metrics.
    */
  def evaluate(iterator: DataSetIterator, metrics: List[zio.nn.EvalMetric]): Try[Map[String, Double]] =
    Try {
      iterator.reset()
      val predBuf = scala.collection.mutable.ArrayBuffer[Float]()
      val labelBuf = scala.collection.mutable.ArrayBuffer[Float]()
      var nOut = -1
      while iterator.hasNext do
        val ds = iterator.next()
        val output = underlying.output(ds.getFeatures)
        val batchSize = ds.getFeatures.shape()(0).toInt
        if nOut < 0 then nOut = output.length().toInt / batchSize
        val outLen = output.length().toInt
        val outArr = new Array[Float](outLen)
        for i <- 0 until outLen do outArr(i) = output.getFloat(i.toLong)
        predBuf ++= outArr
        val lblLen = ds.getLabels.length().toInt
        val lblArr = new Array[Float](lblLen)
        for i <- 0 until lblLen do lblArr(i) = ds.getLabels.getFloat(i.toLong)
        labelBuf ++= lblArr

      val predDouble = predBuf.toArray.map(_.toDouble)
      val actualDouble = labelBuf.toArray.map(_.toDouble)
      if nOut <= 1 then
        metrics.map(m => m.name -> m.compute(predDouble, actualDouble)).toMap
      else
        val pred2D = predDouble.grouped(nOut).toArray
        val act2D  = actualDouble.grouped(nOut).toArray
        val predClass = pred2D.map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
        val actClass  = act2D.map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
        metrics.map(m => m.name -> m.compute(predClass, actClass)).toMap
    }

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

  def toOnnx(model: MultiLayerNetwork, path: Path): Try[Unit] =
    Dl4jToOnnx.saveToFile(model, path)

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
