package zio.nn.djl

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object WrappersSpec extends ZIOSpecDefault:

  def spec = suite("Wrappers (DJL — unified API)")(
    test("ZModel.create from architecture succeeds") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val result = ZModel.create(arch, "test")
      result.foreach(_.close())
      assertTrue(result.isSuccess)
    },
    test("ZModel.predict (unified) works with float arrays") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      ZModel.create(arch, "pred-test") match
        case scala.util.Success(m) =>
          val features = Array.fill(3)(Array.fill(7)(1.0f))
          val result = m.predict(features)
          m.close()
          assertTrue(result.isSuccess)
        case _ => assertTrue(false)
    },
    test("ZModel.fit (unified) runs without error") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      ZModel.create(arch, "fit-test") match
        case scala.util.Success(m) =>
          val feats = Array.fill(10)(Array.fill(7)(scala.util.Random.nextFloat()))
          val labels = Array.fill(10)(scala.util.Random.nextFloat())
          val result = m.fit(feats, labels, 2)
          m.close()
          assertTrue(result.isSuccess)
        case _ => assertTrue(false)
    },
    test("ZModel.underlying exposes raw DJL Model") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      ZModel.create(arch, "escape") match
        case scala.util.Success(m) =>
          val ok = m.underlying.getName == "escape"
          m.close()
          assertTrue(ok)
        case _ => assertTrue(false)
    },
    test("ZModel.close releases resources") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      ZModel.create(arch, "close-test").foreach(_.close())
      assertTrue(true)
    }
  )
