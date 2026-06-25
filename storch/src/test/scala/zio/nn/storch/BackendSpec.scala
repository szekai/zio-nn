package zio.nn.storch

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object BackendSpec extends ZIOSpecDefault:

  def spec = suite("Backend (storch)")(
    test("compile Sequential model with Dense produces non-null module") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("compile LSTM layer") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.LSTM(7, 64, ActivationFn.Tanh), LayerDef.Output(64, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("compile GRU layer") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(AnyLayer.Advanced(AdvancedLayerDef.GRU(7, 64)), AnyLayer.Standard(LayerDef.Output(64, 1)))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("compile MultiHeadAttention layer") {
      val arch = ModelDef.Sequential(SequentialDef(64,
        List(AnyLayer.Advanced(AdvancedLayerDef.MultiHeadAttention(64, 4)),
             AnyLayer.Standard(LayerDef.Dense(64, 10, ActivationFn.ReLU)),
             AnyLayer.Standard(LayerDef.Output(10, 1, LossFn.MSE)))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("all activations compile") {
      val acts = List(ActivationFn.Tanh, ActivationFn.ReLU, ActivationFn.Sigmoid,
                      ActivationFn.Identity, ActivationFn.LeakyReLU())
      assertTrue(acts.forall { act =>
        val arch = ModelDef.Sequential(SequentialDef(7,
          List(LayerDef.Dense(7, 5, act), LayerDef.Output(5, 1, LossFn.MSE))))
        Backend.compile(arch) != null
      })
    },
    test("BatchNorm compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.BatchNorm(10),
             LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("Dropout compiles") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Dropout(0.3),
             LayerDef.Output(10, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("LayerNorm compiles") {
      val arch = ModelDef.Sequential(SequentialDef(64,
        List(LayerDef.Dense(64, 64, ActivationFn.ReLU), LayerDef.LayerNorm(64),
             LayerDef.Output(64, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("Conv2D and MaxPool2D compile") {
      val arch = ModelDef.Sequential(SequentialDef(1,
        List(
          AnyLayer.Standard(LayerDef.Conv2D(1, 16, (3, 3), (1, 1), ActivationFn.ReLU)),
          AnyLayer.Standard(LayerDef.MaxPool2D((2, 2))),
          AnyLayer.Standard(LayerDef.Flatten),
          AnyLayer.Standard(LayerDef.Dense(16 * 13 * 13, 10, ActivationFn.ReLU)),
          AnyLayer.Standard(LayerDef.Output(10, 1, LossFn.MSE))),
        convInput = Some(ConvInput(28, 28, 1))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("TransformerEncoder compiles") {
      val arch = ModelDef.Sequential(SequentialDef(
        inputSize = 128,
        layers = List(
          AnyLayer.Advanced(AdvancedLayerDef.TransformerEncoder(128, 4, 512, 2, 0.1)),
          AnyLayer.Standard(LayerDef.Output(128, 1)))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("Embedding compiles") {
      val arch = ModelDef.Sequential(SequentialDef(1,
        List(LayerDef.Embedding(1000, 128), LayerDef.Output(128, 1, LossFn.MSE))))
      assertTrue(Backend.compile(arch) != null)
    },
    test("FunctionalDef DAG compiles and runs forward pass") {
      val arch = ModelDef.Functional(FunctionalDef(
        inputs = List("in"),
        layers = Map(
          "d1"  -> LayerDef.Dense(7, 10, ActivationFn.ReLU),
          "out" -> LayerDef.Output(10, 1, LossFn.MSE)),
        connections = List(("in", "d1"), ("d1", "out")),
        outputs = List("out")
      ))
      val module = Backend.compile(arch)
      val input  = torch.rand[torch.Float32](4, 7)
      val output = module(input)
      val shape  = output.shape
      assertTrue(module != null, shape(0).toInt == 4, shape(1).toInt == 1)
    },
    test("Dense forward pass produces expected output shape") {
      val arch = ModelDef.Sequential(SequentialDef(7,
        List(LayerDef.Dense(7, 10, ActivationFn.ReLU), LayerDef.Output(10, 1, LossFn.MSE))))
      val module = Backend.compile(arch)
      val input  = torch.rand[torch.Float32](4, 7)
      val output = module(input)
      val shape  = output.shape
      assertTrue(shape(0).toInt == 4, shape(1).toInt == 1)
    },
    test("LSTM forward pass runs without error") {
      val arch = ModelDef.Sequential(SequentialDef(16,
        List(LayerDef.LSTM(16, 32, ActivationFn.Tanh), LayerDef.Output(32, 1, LossFn.MSE))))
      val module = Backend.compile(arch)
      val input  = torch.rand[torch.Float32](4, 8, 16)
      val output = module(input)
      val shape  = output.shape
      // Output shape: (batch, seq_len, 1) after Output(32 -> 1)
      assertTrue(shape(0).toInt == 4, shape(1).toInt == 8, shape(2).toInt == 1)
    }
  )
