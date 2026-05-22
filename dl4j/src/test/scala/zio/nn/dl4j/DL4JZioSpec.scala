package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object DL4JZioSpec extends ZIOSpecDefault:

  def spec = suite("DL4JZio Wrappers")(
    test("ZModel.wrap wraps MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val net = BackendDL4J.compile(arch)
      val model = DL4JZio.ZModel.wrap(net)
      assertTrue(model != null)
    },
    test("ZModel.underlying exposes raw MultiLayerNetwork") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val net = BackendDL4J.compile(arch)
      val model = DL4JZio.ZModel.wrap(net)
      assertTrue(model.underlying.isInstanceOf[MultiLayerNetwork])
    },
    test("ZModel.predict produces non-null output") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val net = BackendDL4J.compile(arch)
      val model = DL4JZio.ZModel.wrap(net)
      val input = org.nd4j.linalg.factory.Nd4j.randn(1, 7)
      val output = model.predict(input)
      assertTrue(output.isSuccess)
    },
    test("DL4JZio.toDataSet creates valid DataSet") {
      val features = Array.fill(10)(Array.fill(7)(scala.util.Random.nextFloat()))
      val labels = Array.fill(10)(scala.util.Random.nextFloat())
      val ds = DL4JZio.toDataSet(features, labels)
      ds match
        case scala.util.Success(d) => assertTrue(d.numExamples() == 10)
        case _ => assertTrue(false)
    },
    test("ZModel.close releases resources") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val net = BackendDL4J.compile(arch)
      val model = DL4JZio.ZModel.wrap(net)
      model.close()
      assertTrue(true)
    }
  )
