package zio.nn.djl

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object BackendSpec extends ZIOSpecDefault:

  def spec = suite("Backend (DJL)")(
    test("compile Sequential model produces non-null DJL Block") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("LSTM layer compiles without error") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.LSTM(7, 64, ActivationFn.Tanh), LayerDef.Output(64, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("BatchNorm layer compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.BatchNorm(10), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("Dropout layer compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Dropout(0.3), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("all activations compile") {
      val acts = List(ActivationFn.Tanh, ActivationFn.ReLU, ActivationFn.Sigmoid, ActivationFn.Identity, ActivationFn.LeakyReLU)
      assertTrue(acts.forall { act =>
        val arch = ModelDef.Sequential(SequentialDef(7, List(LayerDef.Dense(7, 5, act), LayerDef.Output(5, 1, LossFn.MSE))))
        Backend.compile(arch) != null
      })
    }
  )
