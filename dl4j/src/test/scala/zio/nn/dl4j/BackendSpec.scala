package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object BackendSpec extends ZIOSpecDefault:

  def spec = suite("Backend (DL4J)")(
    test("compile produces MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch).isInstanceOf[MultiLayerNetwork])
    },
    test("layer count is correct") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 32, ActivationFn.ReLU), LayerDef.Dense(32, 16, ActivationFn.Tanh), LayerDef.Output(16, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch).getnLayers == 3)
    },
    test("LSTM compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.LSTM(7, 64, ActivationFn.Tanh), LayerDef.Output(64, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("BatchNorm compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.BatchNorm(10), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("Dropout compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Dropout(0.3), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("Adam optimizer is default") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE)),
        optimizer = OptimizerDef.Adam(0.001)))
      assertTrue(Backend.compile(arch) != null)
    }
  )
