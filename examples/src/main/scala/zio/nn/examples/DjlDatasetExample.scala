package zio.nn.examples

import zio.*
import zio.nn.djl.ZModel
import zio.nn.djl.Backend
import zio.nn.djl.zioApi.*
import zio.nn.dsl.*
import zio.nn.{EvalMetric, FitResult}
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDManager
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Dataset

/**
 * Example: DJL Dataset training and evaluation.
 *
 * Trains an XOR network using a DJL [[ai.djl.training.dataset.Dataset]]
 * and evaluates it — demonstrating the `fitDatasetZ` / `evaluateDatasetZ`
 * APIs added in the RNN / Iterator / Dataset integration feature.
 *
 * Run:
 * {{{
 *   sbt "examples/runMain zio.nn.examples.DjlDatasetExample"
 * }}}
 */
object DjlDatasetExample extends ZIOAppDefault:

  /** XOR truth table. */
  private val xorInputs  = Array(Array(0f, 0f), Array(0f, 1f), Array(1f, 0f), Array(1f, 1f))
  private val xorLabels   = Array(0f, 1f, 1f, 0f)

  private def buildDataset(ndm: NDManager): Dataset =
    val data  = ndm.create(xorInputs.flatten).reshape(4, 2)
    val labels = ndm.create(xorLabels.map(Array[Float](_).toArray).flatten).reshape(4, 1)
    ArrayDataset.Builder()
      .setData(data).optLabels(labels)
      .build()

  override def run: ZIO[Any, Any, Any] =
    ZIO.scoped {
      // DJL Dataset construction requires an NDManager
      ZIO.fromAutoCloseable(ZIO.attempt(NDManager.newBaseManager())).flatMap { implicit ndm =>
        for
          arch = Sequential(2)(Dense(4, Tanh), Output(1, MSE)).build

          // Create model
          model <- create(arch)

          // Train via Dataset
          trainDs     = buildDataset(ndm)
          trainResult <- model.fitDatasetZ(trainDs, epochs = 200, batchSize = 2, lr = 0.1f)
          _           <- ZIO.logInfo(s"Train complete — loss: ${trainResult.loss}")

          // Evaluate via Dataset
          evalDs <- ZIO.attempt(buildDataset(ndm))
          metrics <- model.evaluateDatasetZ(evalDs, batchSize = 2, List(EvalMetric.Accuracy()))
          _      <- ZIO.logInfo(s"Evaluation: $metrics")

        yield ()
      }
    }
