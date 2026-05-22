package zio.nn

import zio.test.*
import zio.test.Assertion.*

object ModelArchitectureSpec extends ZIOSpecDefault:

  def spec = suite("ModelArchitecture DSL")(
    test("LayerDef.LSTM produces correct structure") {
      val LayerDef.LSTM(nIn, nOut, act, drop) = LayerDef.LSTM(nIn = 7, nOut = 64, activation = ActivationFn.Tanh)
      assertTrue(nIn == 7, nOut == 64, act == ActivationFn.Tanh, drop == 0.0)
    },
    test("LayerDef.Dense with default activation is ReLU") {
      val LayerDef.Dense(_, _, act) = LayerDef.Dense(nIn = 64, nOut = 32)
      assertTrue(act == ActivationFn.ReLU)
    },
    test("LayerDef.Output defaults to MSE loss and Identity activation") {
      val LayerDef.Output(_, _, loss, act) = LayerDef.Output(nIn = 32, nOut = 1)
      assertTrue(loss == LossFn.MSE, act == ActivationFn.Identity)
    },
    test("SequentialDef constructs with layers in order") {
      val arch = SequentialDef(
        inputSize = 7,
        layers = List(
          LayerDef.LSTM(7, 64, ActivationFn.Tanh),
          LayerDef.Dense(64, 32, ActivationFn.ReLU),
          LayerDef.Output(32, 1, LossFn.MSE)
        ),
        optimizer = OptimizerDef.Adam(0.001),
        seed = 42L
      )
      assertTrue(
        arch.layers.size == 3,
        arch.layers.head.isInstanceOf[LayerDef.LSTM],
        arch.layers.last.isInstanceOf[LayerDef.Output],
        arch.seed == 42L
      )
    },
    test("ModelDef.Sequential wraps SequentialDef") {
      val arch = SequentialDef(7, List(LayerDef.Dense(7, 10, ActivationFn.ReLU)))
      val model = ModelDef.Sequential(arch)
      assertTrue(model match
        case ModelDef.Sequential(a) => a.layers.size == 1
        case _ => false
      )
    },
    test("ActivationFn covers all common activations") {
      val all = ActivationFn.values
      assertTrue(all.size >= 5) // Tanh, ReLU, Sigmoid, Softmax, Identity, LeakyReLU
    },
    test("LossFn covers regression and classification losses") {
      val losses = LossFn.values
      assertTrue(losses.contains(LossFn.MSE), losses.contains(LossFn.MAE))
    },
    test("OptimizerDef supports Adam, SGD, RMSprop") {
      val adam = OptimizerDef.Adam(0.001)
      val sgd = OptimizerDef.SGD(0.01)
      val rms = OptimizerDef.RMSprop(0.001)
      assertTrue(adam.toString.contains("Adam"), sgd.toString.contains("SGD"), rms.toString.contains("RMSprop"))
    },
    test("LayerDef.Dropout accepts rate parameter") {
      val LayerDef.Dropout(rate) = LayerDef.Dropout(0.5)
      assertTrue(rate == 0.5)
    },
    test("LayerDef.BatchNorm carries nIn") {
      val LayerDef.BatchNorm(nIn) = LayerDef.BatchNorm(nIn = 64)
      assertTrue(nIn == 64)
    }
  )
