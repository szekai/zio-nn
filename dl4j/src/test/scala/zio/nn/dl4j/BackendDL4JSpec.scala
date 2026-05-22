package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object BackendDL4JSpec extends ZIOSpecDefault:

  def spec = suite("BackendDL4J")(
    test("compile Sequential model produces MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 10, ActivationFn.ReLU),
          LayerDef.Output(10, 1, LossFn.MSE)
        )
      ))
      val net = BackendDL4J.compile(arch)
      assertTrue(net.isInstanceOf[MultiLayerNetwork])
    },
    test("compiled network has correct number of layers") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 32, ActivationFn.ReLU),
          LayerDef.Dense(32, 16, ActivationFn.Tanh),
          LayerDef.Output(16, 1, LossFn.MSE)
        )
      ))
      val net = BackendDL4J.compile(arch)
      assertTrue(net.getnLayers == 3)
    },
    test("LSTM layer compiles to valid network") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.LSTM(7, 64, ActivationFn.Tanh),
          LayerDef.Output(64, 1, LossFn.MSE)
        )
      ))
      val net = BackendDL4J.compile(arch)
      assertTrue(net != null)
    },
    test("BatchNorm layer compiles") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 10, ActivationFn.ReLU),
          LayerDef.BatchNorm(10),
          LayerDef.Output(10, 1, LossFn.MSE)
        )
      ))
      val net = BackendDL4J.compile(arch)
      assertTrue(net != null)
    },
    test("Dropout layer compiles") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 10, ActivationFn.ReLU),
          LayerDef.Dropout(0.3),
          LayerDef.Output(10, 1, LossFn.MSE)
        )
      ))
      val net = BackendDL4J.compile(arch)
      assertTrue(net != null)
    },
    test("optimizer selection: Adam is default") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.Dense(7, 5, ActivationFn.ReLU),
          LayerDef.Output(5, 1, LossFn.MSE)
        ),
        optimizer = OptimizerDef.Adam(0.001)
      ))
      val net = BackendDL4J.compile(arch)
      assertTrue(net != null)
    }
  )
