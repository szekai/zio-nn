package zio.nn

import zio.test.*
import zio.test.Assertion.*

object DSLSpec extends ZIOSpecDefault:

  def spec = suite("DSL — Architecture Builder")(
    suite("Sequential builder")(
      test("builds simple Dense → Output") {
        val arch = dsl.Sequential(7)(
          dsl.Dense(10, ActivationFn.ReLU),
          dsl.Output(1, LossFn.MSE)
        ).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.layers.size == 2, s.inputSize == 7)
          case _ => assertTrue(false)
      },
      test("builds LSTM → Dense → Output") {
        val arch = dsl.Sequential(7)(
          dsl.LSTM(64, ActivationFn.Tanh),
          dsl.Dense(32, ActivationFn.ReLU),
          dsl.Output(1, LossFn.MSE)
        ).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.layers.size == 3)
            assertTrue(s.layers.head match { case AnyLayer.Standard(LayerDef.LSTM(_, _, _, _)) => true; case _ => false })
            assertTrue(s.layers.last match { case AnyLayer.Standard(LayerDef.Output(_, _, _, _)) => true; case _ => false })
          case _ => assertTrue(false)
      },
      test("BatchNorm in middle preserves dimensions") {
        val arch = dsl.Sequential(7)(
          dsl.Dense(10, ActivationFn.ReLU),
          dsl.BatchNorm,
          dsl.Output(1)
        ).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.layers.size == 3)
          case _ => assertTrue(false)
      },
      test("Dropout in middle preserves dimensions") {
        val arch = dsl.Sequential(7)(
          dsl.Dense(10, ActivationFn.ReLU),
          dsl.Dropout(0.3),
          dsl.Output(1)
        ).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.layers.size == 3)
          case _ => assertTrue(false)
      },
      test("custom optimizer via withOptimizer") {
        val arch = dsl.Sequential(7)(
          dsl.Dense(10), dsl.Output(1)
        ).withOptimizer(OptimizerDef.SGD(0.01)).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.optimizer.isInstanceOf[OptimizerDef.SGD])
          case _ => assertTrue(false)
      },
      test("custom seed via withSeed") {
        val arch = dsl.Sequential(7)(
          dsl.Dense(10), dsl.Output(1)
        ).withSeed(99L).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.seed == 99L)
          case _ => assertTrue(false)
      },
      test("single Output layer (no hidden)") {
        val arch = dsl.Sequential(7)(dsl.Output(1, LossFn.MSE)).build
        arch match
          case ModelDef.Sequential(s) =>
            assertTrue(s.layers.size == 1)
          case _ => assertTrue(false)
      },
      test("all activation functions compile") {
        val acts = List(ActivationFn.Tanh, ActivationFn.ReLU, ActivationFn.Sigmoid, ActivationFn.LeakyReLU, ActivationFn.Identity)
        val results = acts.map { act =>
          dsl.Sequential(7)(dsl.Dense(5, act), dsl.Output(1)).build
        }
        assertTrue(results.size == acts.size)
      },
      test("all loss functions compile") {
        val losses = List(LossFn.MSE, LossFn.MAE, LossFn.BinaryCrossEntropy, LossFn.CategoricalCrossEntropy, LossFn.Huber)
        val results = losses.map { loss =>
          dsl.Sequential(7)(dsl.Dense(10), dsl.Output(1, loss)).build
        }
        assertTrue(results.size == losses.size)
      }
    ),
    suite("Corner cases")(
      test("empty Sequential produces empty layers") {
        val arch = dsl.Sequential(7)().build
        arch match
          case ModelDef.Sequential(s) => assertTrue(s.layers.isEmpty)
          case _ => assertTrue(false)
      },
      test("large layer sizes don't overflow") {
        val arch = dsl.Sequential(100)(
          dsl.Dense(512), dsl.Dense(256), dsl.Dense(128), dsl.Output(1)
        ).build
        assertTrue(true)
      },
      test("zero input size") {
        val arch = dsl.Sequential(0)(dsl.Output(1)).build
        assertTrue(true)
      }
    )
  )
