package zio.nn.examples

import zio.*

/**
 * Entry point listing all available examples in this module.
 *
 * Run from the project root:
 * {{{
 *   sbt "examples/run"
 * }}}
 * then select the example to run, or run directly:
 * {{{
 *   sbt "examples/runMain zio.nn.examples.Dl4jIteratorExample"
 *   sbt "examples/runMain zio.nn.examples.DjlDatasetExample"
 *   sbt "examples/runMain zio.nn.examples.RnnTimeStepExample"
 * }}}
 */
object Main extends ZIOAppDefault:
  override def run: ZIO[Any, Any, Any] =
    for
      _ <- Console.printLine("zio-nn examples")
      _ <- Console.printLine("  [1] Dl4jIteratorExample  — DL4J DataSetIterator training + eval (XOR)")
      _ <- Console.printLine("  [2] DjlDatasetExample   — DJL Dataset training + eval (XOR)")
      _ <- Console.printLine("  [3] RnnTimeStepExample  — RNN time-step streaming inference")
      _ <- Console.printLine("")
      _ <- Console.printLine("Run directly: sbt 'examples/runMain zio.nn.examples.<ExampleName>'")
    yield ()
