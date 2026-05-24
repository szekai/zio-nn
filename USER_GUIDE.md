# zio-nn User Guide

## Architecture

zio-nn has three layers that work together to let you write framework-agnostic neural network code:

```
┌─────────────────────────────────────────────────────────┐
│  Your Code                                              │
│  import zio.nn.*                                        │
│  val arch = Sequential(7)(LSTM(64), Dense(32), Output(1))│
│  val model = ZModel.create(arch, "m").get                │
│  model.predict(features)                                 │
├─────────────────────────────────────────────────────────┤
│  DSL Layer (core)                                       │
│  ModelDef → SequentialDef → LayerDef → ActivationFn     │
│  Pure data — zero framework deps                        │
├─────────────────────────────────────────────────────────┤
│  Backend Layer (djl / dl4j)                             │
│  Backend.compile(ModelDef) → Block / MultiLayerNetwork  │
│  ZModel wraps native model objects                      │
├─────────────────────────────────────────────────────────┤
│  Native Framework                                       │
│  ai.djl.Model (PyTorch/ONNX/TF) | MultiLayerNetwork     │
└─────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Define architecture as data

You define your model structure using the DSL. This produces a `ModelDef` — a pure data structure with no framework dependencies:

```scala
import zio.nn.dsl.*

val arch = Sequential(7)(
  LSTM(64, Tanh),
  Dense(32, ReLU),
  Output(1, MSE)
).build
// arch: ModelDef = Sequential(SequentialDef(7, List(LSTM(...), Dense(...), Output(...))))
```

`ModelDef` is an `enum` with two variants: `Sequential` and `Functional`. It can be serialized to JSON, stored in config files, or constructed programmatically.

### 2. Compile to a backend

The `Backend` object in each module compiles `ModelDef` into a framework-specific model:

```scala
// DJL: ModelDef → ai.djl.nn.Block
val block: ai.djl.nn.Block = Backend.compile(arch)

// DL4J: ModelDef → MultiLayerNetwork
val net: MultiLayerNetwork = Backend.compile(arch)
```

Each layer type (`LSTM`, `Dense`, `Output`, etc.) has a corresponding implementation in both backends. The compiler iterates the layer list and builds the framework-native equivalent.

### 3. Wrap in ZModel

`ZModel` wraps the native model object and provides the unified API:

```scala
// DJL
val model = ZModel.create(arch, "my-model").get
// Under the hood: Backend.compile(arch) → Model.newInstance() → ZModel

// DL4J
val model = ZModel.create(arch)
// Under the hood: Backend.compile(arch) → ZModel.wrap(net)
```

`ZModel` manages:
- **NDManager** (DJL) — native memory allocation
- **Predictor** (DJL) — inference session
- **Trainer** (DJL) — training loop
- **Resource lifecycle** — `.close()` releases everything

### 4. Predict and Train

The unified API converts `Array[Array[Float]]` to/from the framework's native tensor format internally:

```
Your Code                    ZModel                   Framework
─────────                    ──────                   ─────────
predict(features)  ───→  NDList(features)   ───→  model.predict()
        ↑                      │                         │
        │              Array[Float]              NDList output
        └──────────────────────┘─────────────────────────┘
```

```scala
// Both backends — identical code
model.predict(features)  // Array[Array[Float]] → Array[Float]
model.fit(features, labels, epochs = 50)  // returns FitResult(loss, epochs)
```

### 5. SLF4J-style re-exports

Each backend module has an `exports.scala` that re-exports its types into the `zio.nn` package:

```scala
// djl/exports.scala
package zio.nn
export zio.nn.djl.{ZModel, Backend}

// dl4j/exports.scala
package zio.nn
export zio.nn.dl4j.{ZModel, Backend}
```

When you write `import zio.nn.*`, Scala resolves to whichever backend JAR is on the classpath. Since only one backend is ever present, there are no conflicts.

## Lifecycle

```
create(arch, name)
  │
  ├── predict(features)  →  Array[Float]
  ├── predict(features)  →  Array[Float]   (reuse model)
  ├── fit(features, labels, 50)  →  FitResult
  ├── save(path)
  │
  └── close()   ← always call this
```

**Resource management options:**

```scala
// Manual (Try-based)
val model = ZModel.create(arch, "m").get
try { model.predict(features) } finally model.close()

// ZIO Scope (auto-close)
import zio.nn.zioApi.*
ZIO.scoped {
  for
    model <- create(arch, "m")
    pred  <- model.predictZ(features)
  yield pred
} // model.close() called automatically
```

## Escape Hatches

When the unified API doesn't cover your use case, bypass to the raw framework:

```scala
// DJL — access raw ai.djl.Model
val rawModel: ai.djl.Model = model.underlying
val block = rawModel.getBlock
block.getChildren.forEach(println)

// Configure multi-GPU (DJL-specific)
import ai.djl.Device
val gpuConfig = new DefaultTrainingConfig(loss)
  .optDevices(Array(Device.gpu(0), Device.gpu(1)))
val trainer = model.underlying.newTrainer(gpuConfig)

// DL4J — access raw MultiLayerNetwork
val rawNet: MultiLayerNetwork = model.underlying
rawNet.getLayers.foreach(println)

// Spark distributed training (DL4J-specific)
val sparkNet = new SparkDl4jMultiLayer(sc, rawNet, trainingMaster)
```

**When to use escape hatches:**
- Multi-GPU training (DJL) or Spark distributed (DL4J)
- Custom layers not in the DSL
- Transfer learning / fine-tuning
- Model import from external formats
- Fine-grained training control

## DSL Layer Reference

### Sequential

```scala
Sequential(inputSize)(
  LSTM(64, Tanh),
  Dense(32, ReLU),
  Output(1, MSE)
).withOptimizer(Adam(0.001)).withSeed(42L).build
```

### Convolutional

```scala
Sequential(3)(                                    // 3 input channels
  Conv2D(32, (3,3)), MaxPool2D((2,2)),            // 32 filters
  Conv2D(64, (3,3)), MaxPool2D((2,2)),            // 64 filters
  Flatten, Dense(128, ReLU), Output(10, Softmax)  // classifier
).build
```

### All layer types

| Layer | DSL | Purpose |
|-------|-----|---------|
| LSTM | `LSTM(64, Tanh)` | Recurrent / sequence |
| Dense | `Dense(32, ReLU)` | Fully connected |
| Output | `Output(1, MSE)` | Final layer (regression or classification) |
| BatchNorm | `BatchNorm` | Normalization |
| Dropout | `Dropout(0.3)` | Regularization |
| Conv2D | `Conv2D(32, (3,3))` | 2D convolution |
| MaxPool2D | `MaxPool2D((2,2))` | Downsampling |
| Flatten | `Flatten` | Flatten spatial → 1D |

### Activations, Losses, Optimizers

```scala
// Activations
Tanh, ReLU, Sigmoid, Softmax, Identity, LeakyReLU

// Losses
MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy, Huber

// Optimizers
Adam(0.001), SGD(0.01), RMSprop(0.001)
```

## ZIO-Native API

```scala
import zio.nn.zioApi.*
ZIO.scoped {
  for
    model <- create(arch, "m")          // ZIO[Scope, Throwable, ZModel]
    pred  <- model.predictZ(features)    // Task[Array[Float]]
    _     <- model.fitZ(feats, labs, 50) // Task[FitResult]
  yield pred
}
```

## ZIO Stream

```scala
// Live prediction stream
priceStream.via(model.predictFlow).runCollect

// Online training stream
dataStream.via(model.fitFlow(epochs = 1)).runDrain
```

## Tensor Operations

```scala
import zio.nn.TensorOps.*
for
  a <- create(Array(Array(1f,2f), Array(3f,4f)))
  b <- create(Array(Array(0.5f,0.5f), Array(0.5f,0.5f)))
  c <- add(a, b)        // element-wise
  d <- matMul(a, b)     // matrix multiply
  e <- toDoubleArray(d)
yield e
```

## Implicit Conversions

Bridge between unified `Array[Float]` and native tensor types:

```scala
import zio.nn.implicits.*
val nd: NDList = features.toNDList      // unified → native
val back: Array[Array[Float]] = nd.toFloatArrays  // native → unified
```

## Backend Swap

Change one line in `build.sbt` — zero code changes:

```scala
"io.github.szekai" %% "zio-nn-djl"  % "0.7.1"  // ← swap
"io.github.szekai" %% "zio-nn-dl4j" % "0.7.1"  // ← zero code changes
```

## Choosing a Backend

| | DJL | DL4J |
|---|-----|------|
| Engine | PyTorch 2.7.1 | JVM-native |
| GPU | Auto-detect | Manual CUDA setup |
| Distributed | No | Spark |
| Python deps | libtorch (auto-downloaded) | None |
| ONNX/TF/XGBoost | ✅ via engine param | No |
| Best for | Cloud GPU, PyTorch ecosystem | On-prem, big data pipelines |
