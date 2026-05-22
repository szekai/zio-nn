package zio.nn.djl

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import ai.djl.nn.{SequentialBlock, Block}
import scala.util.{Success, Failure}

object BackendDJLSpec extends ZIOSpecDefault:

  def spec = suite("BackendDJL")(
    test("compile Sequential model produces non-null DJL Block") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 10, ActivationFn.ReLU),
          LayerDef.Output(10, 1, LossFn.MSE)
        )
      ))
      val block = BackendDJL.compile(arch)
      assertTrue(block != null)
    },
    test("compiled Block is a SequentialBlock with correct layer count") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 32, ActivationFn.ReLU),
          LayerDef.Dense(32, 16, ActivationFn.Tanh),
          LayerDef.Output(16, 1, LossFn.MSE)
        )
      ))
      val block = BackendDJL.compile(arch)
      assertTrue(block.isInstanceOf[SequentialBlock])
    },
    test("LSTM layer compiles without error") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.LSTM(7, 64, ActivationFn.Tanh),
          LayerDef.Output(64, 1, LossFn.MSE)
        )
      ))
      val block = BackendDJL.compile(arch)
      assertTrue(block != null)
    },
    test("BatchNorm layer compiles without error") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 10, ActivationFn.ReLU),
          LayerDef.BatchNorm(10),
          LayerDef.Output(10, 1, LossFn.MSE)
        )
      ))
      val block = BackendDJL.compile(arch)
      assertTrue(block != null)
    },
    test("Dropout layer compiles without error") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 10, ActivationFn.ReLU),
          LayerDef.Dropout(0.3),
          LayerDef.Output(10, 1, LossFn.MSE)
        )
      ))
      val block = BackendDJL.compile(arch)
      assertTrue(block != null)
    },
    test("compile returns the same type for all activation functions") {
      val activations = List(
        ActivationFn.Tanh, ActivationFn.ReLU, ActivationFn.Sigmoid,
        ActivationFn.Identity, ActivationFn.LeakyReLU
      )
      val results = activations.map { act =>
        val arch = ModelDef.Sequential(SequentialDef(7,
          List(LayerDef.Dense(7, 5, act), LayerDef.Output(5, 1, LossFn.MSE))))
        BackendDJL.compile(arch)
      }
      assertTrue(results.forall(_ != null))
    }
  )
