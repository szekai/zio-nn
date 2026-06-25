package zio.nn.storch

import zio.*
import zio.stream.*
import zio.nn.{FitResult, LossFn, ModelDef, OptimizerDef}
import scala.concurrent.duration.Duration as ScalaDuration
import java.time.{Duration => JavaDuration}
import torch.*

/** ZIO-native wrappers for the storch ZModel API.
  *
  * Provides ZIO Task, Stream, and timed variants of all ZModel operations,
  * mirroring the same pattern used by the DJL and DL4J backends.
  *
  * {{{
  *   import zio.nn.storch.zioApi.*
  *
  *   ZIO.scoped {
  *     for
  *       model <- create(arch)
  *       pred  <- model.predictZ(input)
  *     yield pred
  *   }
  * }}}
  */
object zioApi:

  extension (model: ZModel)

    // ── Core ZIO wrappers ──────────────────────────────────────────────────

    def predictZ(input: Tensor[Float32]): Task[Tensor[Float32]] =
      ZIO.attempt(model.predict(input).get)

    def fitZ(
      modelDef: ModelDef,
      inputs: Tensor[Float32],
      targets: Tensor[Float32],
      optimizerDef: OptimizerDef,
      loss: LossFn,
      epochs: Int,
      batchSize: Int
    ): Task[FitResult] =
      ZIO.attempt(model.fit(modelDef, inputs, targets, optimizerDef, loss, epochs, batchSize).get)

    def saveZ(path: String): Task[Unit] =
      ZIO.attempt(model.save(path).get)

    def loadZ(path: String): Task[Unit] =
      ZIO.attempt(model.load(path).get)

    def closeZ: Task[Unit] =
      ZIO.attempt(model.close().get)

    def predictIntZ(input: Tensor[Float32]): Task[Array[Int]] =
      ZIO.attempt(model.predictInt(input).get)

    def predictDoubleZ(input: Tensor[Float32]): Task[Array[Double]] =
      ZIO.attempt(model.predictDouble(input).get)

    // ── ZIO Stream wrappers ────────────────────────────────────────────────

    /** Streaming prediction — maps each input tensor through predictZ.
      *
      * {{{
      *   val results: ZStream[Any, Throwable, Tensor[Float32]] =
      *     model.predictFlow(tensors)
      * }}}
      */
    def predictFlow(inputs: => Iterable[Tensor[Float32]]): ZStream[Any, Throwable, Tensor[Float32]] =
      ZStream.fromIterable(inputs).mapZIO(model.predictZ)

    /** Streaming training — maps each (inputs, targets) pair through fitZ.
      *
      * {{{
      *   val losses: ZStream[Any, Throwable, FitResult] =
      *     model.fitFlow(data, arch, Adam(), MSE, epochs = 10, batchSize = 32)
      * }}}
      */
    def fitFlow(
      data: => Iterable[(Tensor[Float32], Tensor[Float32])],
      modelDef: ModelDef,
      optimizerDef: OptimizerDef,
      loss: LossFn,
      epochs: Int,
      batchSize: Int
    ): ZStream[Any, Throwable, FitResult] =
      ZStream.fromIterable(data).mapZIO { case (inputs, targets) =>
        model.fitZ(modelDef, inputs, targets, optimizerDef, loss, epochs, batchSize)
      }

    // ── Timed variants ─────────────────────────────────────────────────────

    /** Predict with timing information.
      *
      * @return a tuple of (prediction, duration)
      */
    def predictTimed(input: Tensor[Float32]): Task[(Tensor[Float32], ScalaDuration)] =
      model.predictZ(input).timed.map { case (duration: JavaDuration, result) =>
        (result, ScalaDuration.fromNanos(duration.toNanos))
      }

    /** Fit with timing information.
      *
      * @return a tuple of (fitResult, duration)
      */
    def fitTimed(
      modelDef: ModelDef,
      inputs: Tensor[Float32],
      targets: Tensor[Float32],
      optimizerDef: OptimizerDef,
      loss: LossFn,
      epochs: Int,
      batchSize: Int
    ): Task[(FitResult, ScalaDuration)] =
      model.fitZ(modelDef, inputs, targets, optimizerDef, loss, epochs, batchSize).timed.map {
        case (duration: JavaDuration, result) =>
          (result, ScalaDuration.fromNanos(duration.toNanos))
      }
