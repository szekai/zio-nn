package zio.nn.dl4j

import zio.nn.*
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.*
import zio.test.*
import zio.test.Assertion.*
import zio.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import scala.collection.JavaConverters.*

/** Simple in-memory iterator over a list of DataSet objects. */
class ListDataSetIterator(datasets: java.util.List[DataSet]) extends DataSetIterator:
  private val iter = datasets.iterator()
  private var _cursor = 0
  def next(): DataSet = { _cursor += 1; iter.next() }
  def hasNext: Boolean = iter.hasNext
  def next(num: Int): DataSet = ???
  override def remove(): Unit = ???
  def inputColumns(): Int = if datasets.isEmpty then 0 else datasets.get(0).getFeatures.columns()
  def totalOutcomes(): Int = if datasets.isEmpty then 0 else datasets.get(0).getLabels.columns()
  def resetSupported(): Boolean = true
  def asyncSupported(): Boolean = false
  def reset(): Unit = { _cursor = 0 }
  def batch(): Int = if datasets.isEmpty then 0 else datasets.get(0).numExamples()
  def cursor(): Int = _cursor
  def numExamples(): Long = datasets.size().toLong
  def setPreProcessor(preProcessor: org.nd4j.linalg.dataset.api.DataSetPreProcessor): Unit = ()
  def getPreProcessor: org.nd4j.linalg.dataset.api.DataSetPreProcessor = null
  def getLabels: java.util.List[String] = java.util.Collections.emptyList()

object RnnAndIteratorSpec extends ZIOSpecDefault:

  def spec = suite("DL4J RNN Stateful & DataSetIterator")(

    // ── RNN Stateful Ops (Try-based) ────────────────────────────────

    test("rnnTimeStep returns INDArray of expected shape") {
      val arch = Sequential(4)(LSTM(8, Tanh), Output(1)).build
      val model = ZModel.create(arch)
      val input = Nd4j.create(Array(
        Array(0.5f, 0.3f, 0.8f, 0.1f),
        Array(0.2f, 0.7f, 0.4f, 0.9f)
      ))
      val result = model.rnnTimeStep(input)
      model.close()
      assertTrue(
        result.isSuccess,
        result.get.shape()(0) == 2,   // batch size preserved
        result.get.shape()(1) == 1    // nOut = 1
      )
    },

    test("rnnClearPreviousState does not throw") {
      val arch = Sequential(4)(LSTM(8, Tanh), Output(1)).build
      val model = ZModel.create(arch)
      val result = model.rnnClearPreviousState()
      model.close()
      assertTrue(result.isSuccess)
    },

    test("two rnnTimeStep calls produce different results (state carries over)") {
      val arch = Sequential(4)(LSTM(8, Tanh), Output(1)).build
      val model = ZModel.create(arch)
      val input = Nd4j.create(Array(Array(0.5f, 0.3f, 0.8f, 0.1f)))
      // First call starts from zero state
      val r1 = model.rnnTimeStep(input).get
      // Second call uses state from first call — same input, different state → different output
      val r2 = model.rnnTimeStep(input).get
      model.close()
      assertTrue(r1.getFloat(0L) != r2.getFloat(0L))
    },

    // ── RNN Stateful Ops (ZIO-based) ────────────────────────────────

    test("rnnTimeStepZ returns INDArray of expected shape") {
      val arch = Sequential(4)(LSTM(8, Tanh), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        val input = Nd4j.create(Array(
          Array(0.5f, 0.3f, 0.8f, 0.1f),
          Array(0.2f, 0.7f, 0.4f, 0.9f)
        ))
        model.rnnTimeStepZ(input).map { result =>
          assertTrue(
            result.shape()(0) == 2,
            result.shape()(1) == 1
          )
        }
      }
    },

    test("rnnClearPreviousStateZ does not throw") {
      val arch = Sequential(4)(LSTM(8, Tanh), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        model.rnnClearPreviousStateZ.as(assertTrue(true))
      }
    },

    // ── DataSetIterator fit/evaluate (Try-based) ────────────────────

    test("fit with DataSetIterator succeeds and returns FitResult") {
      val arch = Sequential(4)(Dense(8, ReLU), Output(1)).build
      val model = ZModel.create(arch)
      val feats = Nd4j.create(Array(
        Array(0.1f, 0.2f, 0.3f, 0.4f),
        Array(0.5f, 0.6f, 0.7f, 0.8f)
      ))
      val labs = Nd4j.create(Array(
        Array(0.5f),
        Array(0.9f)
      ))
      val ds = new DataSet(feats, labs)
      val iterator = new ListDataSetIterator(java.util.Arrays.asList(ds))
      val result = model.fit(iterator, 3)
      model.close()
      assertTrue(result.isSuccess, result.get.epochs == 3)
    },

    test("evaluate with DataSetIterator returns accuracy metric") {
      val arch = Sequential(4)(Dense(8, ReLU), Output(3, MAE, Softmax)).build
      val model = ZModel.create(arch)
      val feats = Nd4j.create(Array(
        Array(0.1f, 0.2f, 0.3f, 0.4f)
      ))
      val labs = Nd4j.create(Array(
        Array(1.0f, 0.0f, 0.0f)
      ))
      val ds = new DataSet(feats, labs)
      val iterator = new ListDataSetIterator(java.util.Arrays.asList(ds))
      val result = model.evaluate(iterator)
      model.close()
      assertTrue(
        result.isSuccess,
        result.get.contains("accuracy"),
        result.get("accuracy") >= 0.0,
        result.get("accuracy") <= 1.0
      )
    },

    // ── DataSetIterator fit/evaluate (ZIO-based) ────────────────────

    test("fitZ with DataSetIterator succeeds") {
      val arch = Sequential(4)(Dense(8, ReLU), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        val feats = Nd4j.create(Array(
          Array(0.1f, 0.2f, 0.3f, 0.4f),
          Array(0.5f, 0.6f, 0.7f, 0.8f)
        ))
        val labs = Nd4j.create(Array(
          Array(0.5f),
          Array(0.9f)
        ))
        val ds = new DataSet(feats, labs)
        val iterator = new ListDataSetIterator(java.util.Arrays.asList(ds))
        model.fitZ(iterator, 2).map { result =>
          assertTrue(result.epochs == 2)
        }
      }
    },

    test("evaluateZ with DataSetIterator returns accuracy") {
      val arch = Sequential(4)(Dense(8, ReLU), Output(3, MAE, Softmax)).build
      zioApi.create(arch).flatMap { model =>
        val feats = Nd4j.create(Array(
          Array(0.1f, 0.2f, 0.3f, 0.4f)
        ))
        val labs = Nd4j.create(Array(
          Array(1.0f, 0.0f, 0.0f)
        ))
        val ds = new DataSet(feats, labs)
        val iterator = new ListDataSetIterator(java.util.Arrays.asList(ds))
        model.evaluateZ(iterator).map { result =>
          assertTrue(result.contains("accuracy"), result("accuracy") >= 0.0)
        }
      }
    }

  )
