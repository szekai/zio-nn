package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object WrappersSpec extends ZIOSpecDefault:

  def spec = suite("Wrappers (DL4J — unified API)")(
    test("ZModel.create from architecture succeeds") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val model = ZModel.create(arch)
      assertTrue(model != null)
    },
    test("ZModel.predict (unified) works with float arrays") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val model = ZModel.create(arch)
      val features = Array.fill(3)(Array.fill(7)(1.0f))
      val result = model.predict(features)
      assertTrue(result.isSuccess)
    },
    test("ZModel.fit (unified) runs without error") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val model = ZModel.create(arch)
      val feats = Array.fill(10)(Array.fill(7)(scala.util.Random.nextFloat()))
      val labels = Array.fill(10)(scala.util.Random.nextFloat())
      val result = model.fit(feats, labels, 2)
      assertTrue(result.isSuccess)
    },
    test("ZModel.underlying exposes raw MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      assertTrue(ZModel.create(arch).underlying.isInstanceOf[MultiLayerNetwork])
    }
  )
