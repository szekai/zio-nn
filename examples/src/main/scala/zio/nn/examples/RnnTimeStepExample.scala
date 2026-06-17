package zio.nn.examples

import zio.*
import zio.nn.dl4j.ZModel
import zio.nn.dl4j.Backend
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.*
import org.nd4j.linalg.factory.Nd4j

/**
 * Example: RNN time-step inference (streaming / generative).
 *
 * Demonstrates `rnnTimeStepZ` and `rnnClearPreviousStateZ` — APIs that
 * allow feeding inputs one time-step at a time while preserving the RNN's
 * internal hidden state between calls. This is the building block for
 * auto-regressive generation (e.g. character-level text generation,
 * real-time sequence prediction).
 *
 * Run:
 * {{{
 *   sbt "examples/runMain zio.nn.examples.RnnTimeStepExample"
 * }}}
 */
object RnnTimeStepExample extends ZIOAppDefault:

  /**
   * Generate a simple sine-wave sequence.
   * Each element is `sin(t / period * 2 * pi)`.
   */
  private def sineSequence(length: Int, period: Double = 8.0): Array[Array[Float]] =
    Array.tabulate(length) { t =>
      Array(math.sin(t / period * 2 * math.Pi).toFloat)
    }

  override def run: ZIO[Any, Any, Any] =
    ZIO.scoped {
      for
        arch = Sequential(1)(LSTM(16, Tanh), Output(1, MSE))
          .withOptimizer(Adam(0.01f)).build

        model <- create(arch)

        // 1. Train on next-step prediction
        fullSeq = sineSequence(64)
        inputs  = fullSeq.init          // all but last
        labels  = fullSeq.tail.map(_.head)  // all but first (shift by 1)
        trainResult <- model.fitZ(inputs, labels, epochs = 100, lr = 0.01f)
        _           <- ZIO.logInfo(s"Training loss: ${trainResult.loss}")

        // 2. Prime the RNN state with a few seed steps
        seed = sineSequence(4)
        _   <- ZIO.foreach(seed)(step => model.rnnTimeStepZ(Nd4j.create(step)))

        // 3. Auto-regressive generation
        generated <- ZIO.foldLeft(1 to 10)(List.empty[Float]) { (acc, _) =>
          model.rnnTimeStepZ(Nd4j.create(Array(Array(acc.lastOption.getOrElse(0f)))))
            .map(out => acc :+ out.getFloat(0L))
        }

        _ <- ZIO.logInfo(s"Generated (seed then auto-regress): ${seed.map(_.head).mkString(", ")} ... ${generated.mkString(", ")}")

        // 4. Reset state for a new sequence
        _ <- model.rnnClearPreviousStateZ
        z  <- model.rnnTimeStepZ(Nd4j.create(Array(Array(0.5f))))
        _ <- ZIO.logInfo(s"After reset, fresh prediction: ${z.getFloat(0L)}")

      yield ()
    }
