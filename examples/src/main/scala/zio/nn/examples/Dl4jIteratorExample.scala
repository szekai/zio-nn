package zio.nn.examples

import zio.*
import zio.nn.dl4j.ZModel
import zio.nn.dl4j.Backend
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.*
import zio.nn.{EvalMetric, FitResult}
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/**
 * Example: DL4J DataSetIterator training and evaluation.
 *
 * Trains an XOR network using a [[org.deeplearning4j.datasets.iterator
 * .utilty.ListDataSetIterator]] and evaluates it — demonstrating
 * the `fitIteratorZ` / `evaluateIteratorZ` APIs added in the RNN /
 * Iterator / Dataset integration feature.
 *
 * Run:
 * {{{
 *   sbt "examples/runMain zio.nn.examples.Dl4jIteratorExample"
 * }}}
 */
object Dl4jIteratorExample extends ZIOAppDefault:

  /** XOR truth table. */
  private val xorInputs  = Array(Array(0f, 0f), Array(0f, 1f), Array(1f, 0f), Array(1f, 1f))
  private val xorLabels   = Array(0f, 1f, 1f, 0f)

  private def buildIterator: DataSetIterator =
    val features = xorInputs.map(Nd4j.create)
    val labels   = xorLabels.map(l => Nd4j.create(Array[Float](l)))
    val datasets = features.zip(labels).map { case (f, l) => DataSet(f, l) }
    ListDataSetIterator(datasets.toList.asJava, 2)

  override def run: ZIO[Any, Any, Any] =
    ZIO.scoped {
      for
        // ── 1. Define architecture ──
        arch = Sequential(2)(Dense(4, Tanh), Output(1, MSE)).build

        // ── 2. Create model ──
        model <- create(arch)

        // ── 3. Train via DataSetIterator ──
        trainIterator = buildIterator
        trainResult   <- model.fitIteratorZ(trainIterator, epochs = 200, lr = 0.1f)
        _             <- ZIO.logInfo(s"Train complete — loss: ${trainResult.loss}")

        // ── 4. Evaluate via DataSetIterator ──
        evalIterator = buildIterator
        metrics      <- model.evaluateIteratorZ(evalIterator, List(EvalMetric.Accuracy()))
        _            <- ZIO.logInfo(s"Evaluation: $metrics")

        // ── 5. Predict to verify ──
        preds <- model.predictZ(xorInputs)
        _     <- ZIO.logInfo(s"Predictions: ${preds.mkString(", ")}")

      yield ()
    }
