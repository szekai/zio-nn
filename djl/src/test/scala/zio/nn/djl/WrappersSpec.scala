package zio.nn.djl

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object WrappersSpec extends ZIOSpecDefault:

  def spec = suite("Wrappers (DJL)")(
    test("ZModel.create succeeds") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val block = Backend.compile(arch)
      val result = ZModel.create(block, "test")
      result.foreach(_.close())
      assertTrue(result.isSuccess)
    },
    test("ZModel.underlying exposes raw ai.djl.Model") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val block = Backend.compile(arch)
      ZModel.create(block, "escape-test") match
        case scala.util.Success(m) =>
          val ok = m.underlying.getName == "escape-test"
          m.close(); assertTrue(ok)
        case _ => assertTrue(false)
    },
    test("ZModel.predictor creates ZPredictor") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      ZModel.create(Backend.compile(arch), "pred-test") match
        case scala.util.Success(m) =>
          val ok = m.predictor().isSuccess
          m.close(); assertTrue(ok)
        case _ => assertTrue(false)
    },
    test("ZPredictor.underlying exposes raw DJL Predictor") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      ZModel.create(Backend.compile(arch), "escape") match
        case scala.util.Success(m) =>
          m.predictor() match
            case scala.util.Success(p) =>
              val ok = p.underlying != null
              p.close(); m.close(); assertTrue(ok)
            case _ => m.close(); assertTrue(false)
        case _ => assertTrue(false)
    },
    test("defaultConfig produces valid config") {
      assertTrue(defaultConfig() != null)
    },
    test("mseLoss is non-null") {
      assertTrue(mseLoss != null)
    }
  )
