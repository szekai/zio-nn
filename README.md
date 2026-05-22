# zio-nn — Neural Network Library for ZIO

**Write once, run on any JVM deep learning framework.**

```scala
// Define architecture as pure data
val arch = Sequential(7,
  LSTM(7  -> 64, Tanh),
  Dense(64 -> 32, ReLU),
  Output(32 -> 1, MSE))

// Compile to DJL (AWS PyTorch engine)
val djlModel = for
  block  <- Try(BackendDJL.compile(arch))
  model  <- ZModel.create(block, "my-model")
yield model

// OR compile to DL4J (Eclipse, no Python deps)
val dl4jNet = BackendDL4J.compile(arch)
val dl4jModel = DL4JZio.ZModel.wrap(dl4jNet)
```

## Modules

| Module | Depends on | Description |
|--------|-----------|-------------|
| `zio-nn-core` | ZIO | Framework-agnostic architecture DSL (`ModelDef`, `LayerDef`, `ActivationFn`) |
| `zio-nn-djl` | core + DJL 0.36 | DJL backend: `BackendDJL` + `DJLZio` wrappers |
| `zio-nn-dl4j` | core + DL4J 1.0.0-M2.1 | DL4J backend: `BackendDL4J` + `DL4JZio` wrappers |

```scala
// sbt
libraryDependencies += "dev.zio" %% "zio-nn-djl" % "0.1.0"  // or zio-nn-dl4j
```

## Quick Start

### 1. Define your architecture

```scala
import zio.nn.*

val myArch = ModelDef.Sequential(SequentialDef(
  inputSize = 10,
  layers = List(
    LayerDef.Dense(10, 64, ActivationFn.ReLU),
    LayerDef.Dropout(0.2),
    LayerDef.Dense(64, 32, ActivationFn.ReLU),
    LayerDef.Output(32, 1, LossFn.MSE)
  ),
  optimizer = OptimizerDef.Adam(learningRate = 0.001)
))
```

### 2. Create a model

```scala
import zio.nn.djl.*

val model = ZModel.create(BackendDJL.compile(myArch), "my-model") match
  case Success(m) => m
  case Failure(e) => throw e
```

### 3. Train

```scala
// Prepare your data (ArrayDataset from companion examples)
val trainSet = ArrayDataset(trainFeatures, trainLabels)
val testSet  = ArrayDataset(testFeatures, testLabels)

model.trainer(DJLZio.defaultConfig(lr = 0.001f)) match
  case Success(trainer) =>
    trainer.initialize(new Shape(1, 10))
    trainer.fit(epochs = 50, trainSet, testSet) match
      case Success(result) => println(s"Loss: ${result.getTrainLoss}")
      case _ => println("Training failed")
    trainer.close()
  case _ => println("Trainer creation failed")
```

### 4. Predict

```scala
model.predictor() match
  case Success(pred) =>
    pred.predict(new NDList(inputTensor)) match
      case Success(output) => println(s"Prediction: ${output.head()}")
      case _ => println("Prediction failed")
    pred.close()
  case _ => println("Predictor creation failed")
```

### 5. Save / Load

```scala
model.save(Path.of("models/my-model"), "my-model")
// Later...
val loaded = ZModel.load(Path.of("models/my-model"), "my-model")
```

---

## Feature Coverage Matrix

### What the DSL covers (80% use case)

| Feature | DJL | DL4J | Notes |
|---------|-----|------|-------|
| LSTM (recurrent) | ✅ | ✅ | Single-layer, configurable state size |
| Dense (fully-connected) | ✅ | ✅ | Any activation |
| Output (regression) | ✅ | ✅ | MSE, MAE, Huber |
| Output (classification) | ✅ | ✅ | Cross-entropy variants |
| Batch Normalization | ✅ | ✅ | |
| Dropout | ⚠️ | ✅ | DJL: configure on Trainer; DL4J: layer config |
| Adam / SGD / RMSprop | ✅ | ✅ | Learning rate, momentum |
| Sequential models | ✅ | ✅ | Layers stacked in order |
| Save / Load | ✅ | ✅ | Framework-native format |
| Train / Predict / Close | ✅ | ✅ | Try-based, manual resource management |

### What requires the escape hatch

These features are framework-specific and intentionally NOT abstracted:

| Feature | How to access | Why not in DSL |
|---------|--------------|----------------|
| Multi-GPU training | `trainer.underlying.setDevices(...)` | DJL-only; DL4J Spark-based |
| Distributed training (Spark) | `model.underlying` → SparkDl4jMultiLayer | DL4J-only; DJL explicitly doesn't support |
| Custom layers / ops | `model.underlying.getBlock()` → custom Block | Infinite possibilities |
| Transfer learning | `model.underlying` → freeze layers | Framework-specific import APIs |
| Model import (PyTorch→DJL) | `Model.newInstance(...)` → `ZModel.wrap(...)` | Different formats per framework |
| Attention / Transformers | Raw DJL Block or DL4J SameDiff | Not yet in DSL; add when needed |
| Fine-grained training control | `trainer.underlying` → customize loop | `EasyTrain` is a convenience; raw loop available |
| Mixed precision (FP16) | `config.optDataType(...)` on raw config | Framework-specific config |
| Early stopping | Manual loop via `trainer.underlying` | DL4J's EarlyStoppingConfig vs DJL's custom listener |

---

## Escape Hatch Pattern

Every wrapper exposes its underlying framework object. Bypass the DSL entirely when needed:

```scala
// DJL — access raw DJL Model
val rawModel: ai.djl.Model = myZModel.underlying
rawModel.getBlock().getChildren().forEach { child =>
  println(s"Layer: ${child.getClass.getSimpleName}")
}

// Configure multi-GPU (DJL-specific)
import ai.djl.Device
val gpuConfig = new DefaultTrainingConfig(loss)
  .optDevices(Array(Device.gpu(0), Device.gpu(1)))
myZModel.underlying.newTrainer(gpuConfig)

// DL4J — access raw MultiLayerNetwork
val rawNet: MultiLayerNetwork = myDL4JModel.underlying
rawNet.getLayers.foreach(println)

// Spark distributed training (DL4J-specific)
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
val sparkNet = new SparkDl4jMultiLayer(sc, rawNet, trainingMaster)
```

**Rule of thumb:** If the DSL has it, use the DSL. If the DSL doesn't have it, use `.underlying`. Never fork the library — the escape hatch is the supported path.

---

## Custom Architectures (No DSL)

If your model can't be expressed in `ModelDef`, skip it entirely:

```scala
// DJL — build raw Block directly
val myBlock = new SequentialBlock()
  .add(Conv2d.builder().setKernelShape(new Shape(3,3)).setFilters(32).build())
  .add(Pool.maxPool2dBlock(new Shape(2,2)))
  .add(Linear.builder().setUnits(10).build())

val model = ZModel.create(myBlock, "cnn")

// DL4J — build raw MultiLayerNetwork directly
val conf = new NeuralNetConfiguration.Builder()
  .list()
  .layer(new ConvolutionLayer.Builder(3,3).nIn(1).nOut(32).build())
  .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2,2)).build())
  .layer(new OutputLayer.Builder(LossFunction.MCXENT).nOut(10).build())
  .build()
val net = new MultiLayerNetwork(conf); net.init()
val model = DL4JZio.ZModel.wrap(net)
```

---

## Why Two Backends?

| | DJL | DL4J |
|---|-----|------|
| **Engine** | PyTorch 2.7.1 native | JVM-native (C++ ND4J) |
| **Training** | GPU + CPU | GPU + CPU + Spark distributed |
| **Python deps** | libtorch (auto-downloaded) | None |
| **Best for** | Cloud GPU servers, PyTorch ecosystem | On-prem JVM shops, big data pipelines |
| **Maintainer** | AWS (150 contributors) | Konduit (1 maintainer) |
| **Release cadence** | Monthly | ~3 years since last stable |

Choose DJL for modern cloud GPU training. Choose DL4J for zero-Python-dependency JVM deployments.

---

## Training Data

The library doesn't ship a data pipeline. Use DJL's `RandomAccessDataset` or DL4J's `DataSetIterator`:

```scala
// DJL — ArrayDataset (included in companion examples)
val ds = ArrayDataset(features, labels)

// DL4J
val ds = DL4JZio.toDataSet(features2d, labels1d)

// Or use any framework-native data pipeline
val mnist = Mnist.builder().setSampling(32, true).build()  // DJL
val recordReader = new CSVRecordReader(...)                  // DL4J DataVec
```

---

## Resource Management

All wrappers use manual `.close()`. Pair with ZIO `Scope` for auto-cleanup:

```scala
// Manual (works everywhere)
val model = ZModel.create(...).get
try { /* use model */ } finally model.close()

// ZIO Scope (auto-close)
ZIO.scoped {
  for
    model <- ZIO.fromTry(ZModel.create(...))
    pred  <- ZIO.fromTry(model.predictor())
    out   <- ZIO.fromTry(pred.predict(input))
  yield out
}
```

---

## Contributing

Add a new backend in 3 steps:
1. Write a `BackendX.scala` interpreter (~50 lines, `ModelDef` → framework Block)
2. Write an `XZio.scala` wrapper (~80 lines, `ZModel`/`ZPredictor`/`ZTrainer`)
3. Add a `zio-nn-x` module to `build.sbt`

---

## License

Apache 2.0
