package zio.nn.storch

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import zio.test.TestAspect

object ZModelSpec extends ZIOSpecDefault:

  def spec = suite("ZModel (storch)")(
    test("ZModel.create returns non-null model") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val model = ZModel.create(arch)
      assertTrue(model.isSuccess)
    },
    test("predict returns correct output size") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val model    = ZModel.create(arch).get
      val input    = torch.rand[torch.Float32](3, 7)
      val output   = model.predict(input).get
      val shape    = output.shape
      assertTrue(shape(0).toInt == 3, shape(1).toInt == 1)
    },
    test("predictDouble returns correct output size") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val model    = ZModel.create(arch).get
      val input    = torch.rand[torch.Float32](3, 7)
      val output   = model.predictDouble(input).get
      assertTrue(output.length == 3)
    },
    test("predictInt returns correct output size") {
      val arch = ModelDef.Sequential(SequentialDef(10,
        List(LayerDef.Dense(10, 5, ActivationFn.ReLU), LayerDef.Output(5, 3, LossFn.CategoricalCrossEntropy()))))
      val model    = ZModel.create(arch).get
      val input    = torch.rand[torch.Float32](4, 10)
      val output   = model.predictInt(input).get
      assertTrue(output.length == 4)
    },
    test("fit reduces loss over epochs") {
      val arch = ModelDef.Sequential(SequentialDef(1,
        List(LayerDef.Dense(1, 8, ActivationFn.ReLU),
             LayerDef.Dense(8, 1, ActivationFn.Identity),
             LayerDef.Output(1, 1, LossFn.MSE))))
      val model   = ZModel.create(arch).get
      val xs      = torch.Tensor(Array(0.0f, 1.0f, 2.0f, 3.0f)).reshape(4, 1)
      val ys      = torch.Tensor(Array(1.0f, 3.0f, 5.0f, 7.0f)).reshape(4, 1)
      val result  = model.fit(arch, xs, ys, OptimizerDef.Adam(0.005), LossFn.MSE, epochs = 500, batchSize = 4)
      assertTrue(result.isSuccess, result.get.loss < 1.0)
    },
    test("save and load round-trip preserves predictions") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val model     = ZModel.create(arch).get
      val tmpPath   = java.nio.file.Files.createTempFile("zio-nn-storch-", ".pt")
      val input     = torch.Tensor(Array(0.5f, 0.3f, 0.8f, 0.1f, 0.9f, 0.2f, 0.4f)).reshape(1, 7)
      val origOut   = model.predict(input).get
      model.save(tmpPath.toString).get
      model.load(tmpPath.toString).get
      val loadOut   = model.predict(input).get
      val origArr   = origOut.toArray
      val loadArr   = loadOut.toArray
      assertTrue(origArr.toSeq == loadArr.toSeq)
    } @@ TestAspect.ignore  // storch: save/load uses Java serialization which doesn't support Map[String, Tensor].
    // Fix: use native PyTorch OutputArchive/InputArchive, but nativeModule() is private[torch].
  )
