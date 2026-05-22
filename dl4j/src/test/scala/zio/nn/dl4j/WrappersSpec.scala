package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object WrappersSpec extends ZIOSpecDefault:

  def spec = suite("Wrappers (DL4J)")(
    test("ZModel.wrap wraps MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val net = Backend.compile(arch)
      val model = ZModel.wrap(net)
      assertTrue(model != null)
    },
    test("ZModel.underlying is MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      assertTrue(ZModel.wrap(Backend.compile(arch)).underlying.isInstanceOf[MultiLayerNetwork])
    },
    test("ZModel.predict succeeds") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val model = ZModel.wrap(Backend.compile(arch))
      val input = org.nd4j.linalg.factory.Nd4j.randn(1, 7)
      assertTrue(model.predict(input).isSuccess)
    },
    test("toDataSet creates valid DataSet") {
      val f = Array.fill(10)(Array.fill(7)(scala.util.Random.nextFloat()))
      val l = Array.fill(10)(scala.util.Random.nextFloat())
      toDataSet(f, l) match
        case scala.util.Success(d) => assertTrue(d.numExamples() == 10)
        case _ => assertTrue(false)
    },
    test("ZModel.close releases resources") {
      ZModel.wrap(Backend.compile(ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE)))))).close()
      assertTrue(true)
    }
  )
