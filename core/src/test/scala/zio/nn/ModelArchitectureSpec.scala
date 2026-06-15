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
          AnyLayer.Standard(LayerDef.LSTM(7, 64, ActivationFn.Tanh)),
          AnyLayer.Standard(LayerDef.Dense(64, 32, ActivationFn.ReLU)),
          AnyLayer.Standard(LayerDef.Output(32, 1, LossFn.MSE))
        ),
        optimizer = OptimizerDef.Adam(0.001),
        seed = 42L
      )
      assertTrue(
        arch.layers.size == 3,
        arch.layers.head.isInstanceOf[AnyLayer.Standard],
        arch.layers.last.isInstanceOf[AnyLayer.Standard],
        arch.seed == 42L
      )
    },
    test("ModelDef.Sequential wraps SequentialDef") {
      val arch = SequentialDef(7, List(AnyLayer.Standard(LayerDef.Dense(7, 10, ActivationFn.ReLU))))
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
    },
    // ── ActivationFn methods ──
    test("ActivationFn.apply returns correct values") {
      assertTrue(
        ActivationFn.ReLU.apply(-1.0) == 0.0,
        ActivationFn.ReLU.apply(3.0) == 3.0,
        ActivationFn.Identity.apply(42.0) == 42.0,
        math.abs(ActivationFn.Sigmoid.apply(0.0) - 0.5) < 1e-10,
        math.abs(ActivationFn.Tanh.apply(0.0)) < 1e-10,
        ActivationFn.LeakyReLU.apply(-1.0) == -0.01,
        ActivationFn.LeakyReLU.apply(2.0) == 2.0
      )
    },
    test("ActivationFn.applyVector softmax sums to 1") {
      val result = ActivationFn.Softmax.applyVector(Array(2.0, 1.0, 0.1))
      assertTrue(
        result.length == 3,
        math.abs(result.sum - 1.0) < 1e-10
      )
    },
    test("ActivationFn.derivative returns correct values") {
      assertTrue(
        ActivationFn.ReLU.derivative(-5.0) == 0.0,
        ActivationFn.ReLU.derivative(3.0) == 1.0,
        ActivationFn.Identity.derivative(99.0) == 1.0,
        ActivationFn.LeakyReLU.derivative(-1.0) == 0.01,
        ActivationFn.LeakyReLU.derivative(5.0) == 1.0
      )
    },
    test("ActivationFn.Sigmoid.derivative matches s*(1-s) formula") {
      val x = 0.5; val s = ActivationFn.Sigmoid.apply(x); val d = ActivationFn.Sigmoid.derivative(x)
      assertTrue(math.abs(d - s * (1.0 - s)) < 1e-10)
    },
    test("ActivationFn.Tanh.derivative matches 1-tanh^2 formula") {
      val x = 0.5; val t = ActivationFn.Tanh.apply(x); val d = ActivationFn.Tanh.derivative(x)
      assertTrue(math.abs(d - (1.0 - t * t)) < 1e-10)
    },
    // ── LossFn methods ──
    test("LossFn.MSE.compute returns 0 for identical arrays") {
      val pred = Array(1.0, 2.0, 3.0); val actual = Array(1.0, 2.0, 3.0)
      assertTrue(LossFn.MSE.compute(pred, actual) == 0.0)
    },
    test("LossFn.MSE.compute returns correct value") {
      assertTrue(math.abs(LossFn.MSE.compute(Array(0.0), Array(1.0)) - 1.0) < 1e-10)
    },
    test("LossFn.MAE.compute returns correct value") {
      assertTrue(LossFn.MAE.compute(Array(1.0, 2.0), Array(3.0, 4.0)) == 2.0)
    },
    test("LossFn.BinaryCrossEntropy.compute returns finite value") {
      val result = LossFn.BinaryCrossEntropy.compute(Array(0.9, 0.1), Array(1.0, 0.0))
      assertTrue(!result.isNaN, !result.isInfinite)
    },
    test("LossFn.Huber.compute is finite") {
      val result = LossFn.Huber.compute(Array(1.0, 2.0), Array(3.0, 4.0))
      assertTrue(!result.isNaN, !result.isInfinite, result > 0)
    },
    // ── EvalMetric methods ──
    test("EvalMetric.Accuracy.compute perfect match") {
      assertTrue(EvalMetric.Accuracy.compute(Array(0.9, 0.1, 0.8), Array(1.0, 0.0, 1.0)) == 1.0)
    },
    test("EvalMetric.Accuracy.compute 50%") {
      assertTrue(EvalMetric.Accuracy.compute(Array(0.9, 0.1), Array(1.0, 1.0)) == 0.5)
    },
    test("EvalMetric.Precision.compute") {
      val p = EvalMetric.Precision().compute(Array(0.9, 0.1, 0.8, 0.2), Array(1.0, 0.0, 1.0, 0.0))
      assertTrue(p >= 0.0 && p <= 1.0)
    },
    test("EvalMetric.Recall.compute") {
      val r = EvalMetric.Recall().compute(Array(0.9, 0.1, 0.8), Array(1.0, 0.0, 1.0))
      assertTrue(r >= 0.0 && r <= 1.0)
    },
    test("EvalMetric.F1.compute perfect") {
      assertTrue(EvalMetric.F1().compute(Array(0.9, 0.1, 0.8), Array(1.0, 0.0, 1.0)) == 1.0)
    }
  )
