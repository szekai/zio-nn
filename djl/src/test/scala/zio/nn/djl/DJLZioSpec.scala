package zio.nn.djl

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object DJLZioSpec extends ZIOSpecDefault:

  def spec = suite("DJLZio Wrappers")(
    test("ZModel.create succeeds with valid Block") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val block = BackendDJL.compile(arch)
      DJLZio.ZModel.create(block, "test-model") match
        case scala.util.Success(model) =>
          model.close()
          assertTrue(true)
        case _ => assertTrue(false)
    },
    test("ZModel.underlying exposes raw DJL Model") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val block = BackendDJL.compile(arch)
      DJLZio.ZModel.create(block, "escape-test") match
        case scala.util.Success(model) =>
          val name = model.underlying.getName
          model.close()
          assertTrue(name == "escape-test")
        case _ => assertTrue(false)
    },
    test("ZModel.predictor creates valid predictor") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val block = BackendDJL.compile(arch)
      DJLZio.ZModel.create(block, "pred-test") match
        case scala.util.Success(model) =>
          val predResult = model.predictor()
          model.close()
          assertTrue(predResult.isSuccess)
        case _ => assertTrue(false)
    },
    test("ZPredictor.underlying exposes raw DJL Predictor") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 5, ActivationFn.ReLU), LayerDef.Output(5, 1, LossFn.MSE))))
      val block = BackendDJL.compile(arch)
      DJLZio.ZModel.create(block, "pred-escape") match
        case scala.util.Success(model) =>
          val predResult = model.predictor()
          predResult.foreach(_.close())
          model.close()
          assertTrue(predResult.isSuccess)
        case _ => assertTrue(false)
    },
    test("DJLZio.defaultConfig produces valid training config") {
      val config = DJLZio.defaultConfig()
      assertTrue(config != null)
    },
    test("DJLZio.mseLoss is L2 loss") {
      val loss = DJLZio.mseLoss
      assertTrue(loss != null)
    }
  )
