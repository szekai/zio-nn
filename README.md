# zio-nn — Neural Network Library for ZIO

**Write once, run on any JVM deep learning framework. Swap the JAR, not the code.**

```scala
// Define architecture — same for both backends
import zio.nn.dsl.*
val arch = Sequential(7)(LSTM(64, Tanh), Dense(32, ReLU), Output(1, MSE)).build

// Create model — same for both backends
import zio.nn.*
val model = ZModel.create(arch, "my-model").get

// Predict / Train — same for both backends
val predictions = model.predict(features).get      // Array[Array[Float]] → Array[Float]
model.fit(features, labels, epochs = 50)             // returns FitResult(loss, epochs)
model.close()
```

**Swap DJL ↔ DL4J by changing one line in `build.sbt` — zero code changes.**

## Modules

| Module | Description |
|--------|-------------|
| `zio-nn-core` | DSL (`dsl.*`), types (`ModelDef`, `LayerDef`, `FitResult`), ConfigLoader, implicits |
| `zio-nn-djl` | ZModel, Backend, zioApi, TensorOps, scope, implicits — PyTorch/ONNX/TF/XGBoost |
| `zio-nn-dl4j` | ZModel, Backend, zioApi, TensorOps, implicits — JVM-native (no Python) |

```scala
// sbt
libraryDependencies += "io.github.szekai" %% "zio-nn-djl" % "0.7.1"  // or zio-nn-dl4j
```

## Quick Start

### 1. Define architecture

```scala
import zio.nn.dsl.*

val arch = Sequential(7)(
  LSTM(64, Tanh),
  Dense(32, ReLU),
  Output(1, MSE)
).withOptimizer(Adam(0.001)).build

// Or with custom seed
val arch2 = Sequential(10)(
  Dense(128, ReLU), Dropout(0.3), Dense(64, ReLU), Output(1, MSE)
).withSeed(99L).build
```

### 2. Create model

```scala
import zio.nn.*

val model = ZModel.create(arch, "my-model") match
  case Success(m) => m
  case Failure(e) => throw e
```

### 3. Train

```scala
val features: Array[Array[Float]] = // your training data
val labels: Array[Float]           = // your labels

model.fit(features, labels, epochs = 50, lr = 0.001f) match
  case Success(result) => println(s"Loss: ${result.getTrainLoss}")
  case _ => println("Training failed")
```

### 4. Predict

```scala
model.predict(features) match
  case Success(result) => println(s"Predictions: ${result.mkString(",")}")
  case _ => println("Prediction failed")
```

### 5. Save / Load

```scala
model.save(Path.of("models/my-model"))
// Later...
val loaded = ZModel.load(Path.of("models/my-model"))
```

---

## DSL Reference

### Layer constructors

| DSL | Expands to |
|-----|-----------|
| `LSTM(64, Tanh)` | `LayerDef.LSTM(nIn=auto, nOut=64, Tanh)` |
| `LSTM(64)` | `LayerDef.LSTM(nIn=auto, nOut=64, Tanh)` (default) |
| `Dense(32, ReLU)` | `LayerDef.Dense(nIn=auto, nOut=32, ReLU)` |
| `Dense(32)` | `LayerDef.Dense(nIn=auto, nOut=32, ReLU)` (default) |
| `Output(1, MSE)` | `LayerDef.Output(nIn=auto, nOut=1, MSE)` |
| `Output(1)` | `LayerDef.Output(nIn=auto, nOut=1, MSE)` (default) |
| `BatchNorm` | `LayerDef.BatchNorm(nIn=auto)` |
| `Dropout(0.3)` | `LayerDef.Dropout(0.3)` |

Input sizes auto-propagate through the chain: `Sequential(7)(LSTM(64), Dense(32), Output(1))` — the compiler resolves 7→64, 64→32, 32→1 automatically.

### Shortcuts

| Shortcut | Type |
|----------|------|
| `Tanh`, `ReLU`, `Sigmoid`, `Softmax` | `ActivationFn` |
| `MSE`, `MAE` | `LossFn` |
| `Adam(0.001)`, `SGD(0.01)`, `RMSprop(0.001)` | `OptimizerDef` |

### Builder chain

```scala
Sequential(7)(
  LSTM(64), Dense(32), Output(1)
).withOptimizer(SGD(0.01)).withSeed(42L).build
```

---

## Feature Coverage Matrix

### Unified API (same for both backends)

| Operation | Signature | Notes |
|-----------|-----------|-------|
| Create | `ZModel.create(arch, name)` | Compiles + initializes internally |
| Predict | `model.predict(features: Array[Array[Float]]): Try[Array[Float]]` | Internal NDList/INDArray conversion |
| Fit | `model.fit(features, labels, epochs, lr)` | Internal dataset creation |
| Save | `model.save(path)` | Framework-native format |
| Load | `ZModel.load(path)` | Auto-detects engine |
| Close | `model.close()` | Releases native resources |

### DSL Coverage (80% use case)

| Feature | DJL | DL4J |
|---------|-----|------|
| LSTM (recurrent) | ✅ | ✅ |
| Dense (fully-connected) | ✅ | ✅ |
| Output (regression/classification) | ✅ | ✅ |
| Batch Normalization | ✅ | ✅ |
| Dropout | ⚠️ identity | ✅ |
| Adam / SGD / RMSprop | ✅ | ✅ |
| Sequential models | ✅ | ✅ |
| Save / Load | ✅ | ✅ |

### Escape Hatches (framework-specific)

| Feature | Access | Why not unified |
|---------|--------|-----------------|
| Multi-GPU | `model.underlying` → engine API | Different config per framework |
| Custom layers | `model.underlying.getBlock()` | Infinite possibilities |
| Distributed (Spark) | `model.underlying` (DL4J only) | DJL explicitly doesn't support |
| Model import | Raw `Model.newInstance()` | Different formats per framework |
| Fine-grained training | `model.predictorRaw()` / `model.trainerRaw()` | Custom loops |
| Conv2D, Transformers | Raw `Block` construction | Not yet in DSL |

---

## Implicit Conversions

Bridge between unified `Array[Float]` and native types when you need the escape hatch:

```scala
// DJL
import zio.nn.djl.implicits.*
// NDManager is managed internally by TensorOps and scope.withNDManager
val nd: NDList = myArrays.toNDList           // unified → native
val back: Array[Array[Float]] = nd.toFloatArrays  // native → unified

// DL4J
import zio.nn.dl4j.implicits.*
val ind: INDArray = myArrays.toINDArray       // unified → native
val back: Array[Float] = ind.toFloatArray     // native → unified
```

| Source | Method | Target | Backend |
|--------|--------|--------|---------|
| `Array[Array[Float]]` | `.toNDList` | `NDList` | DJL |
| `Array[Float]` | `.toNDArray` | `NDArray` | DJL |
| `NDList` | `.toFloatArrays` | `Array[Array[Float]]` | DJL |
| `NDArray` | `.toFloatArray` | `Array[Float]` | DJL |
| `Array[Array[Float]]` | `.toINDArray` | `INDArray` | DL4J |
| `Array[Float]` | `.toINDArray` | `INDArray` | DL4J |
| `INDArray` | `.toFloatArrays` | `Array[Array[Float]]` | DL4J |
| `INDArray` | `.toFloatArray` | `Array[Float]` | DL4J |

---

## Resource Management

```scala
// Manual (Try-based)
val model = ZModel.create(arch, "m").get
try { /* use model */ } finally model.close()

// ZIO-native (Scope-based, v0.3.0+)
import zio.nn.zioApi.*
ZIO.scoped {
  for
    model <- create(arch, "m")           // auto-closed by Scope
    pred  <- model.predictZ(features)
  yield pred
}

// DJL TensorOps scope helper (v0.4.1)
import zio.nn.scope.withNDManager
import zio.nn.TensorOps.*
withNDManager {
  for
    a <- createDouble1D(data)   // explicit NDManager lifecycle
    b <- add(a, a)
  yield b
} // NDManager auto-closed

// Without scope.withNDManager:
// DJL uses an internal base manager with per-call sub-managers — safe for all workloads.
```

## FitResult (v0.4.1)

`model.fit()` returns `Try[FitResult]` across all backends:

```scala
model.fit(features, labels, epochs = 50) match
  case Success(FitResult(loss, epochs)) => println(s"Loss: $loss after $epochs epochs")
  case _ => println("Training failed")
```

## Multi-Input / FunctionalDef

Both backends support `FunctionalDef` for multi-input, skip-connection, and DAG architectures:

```scala
val arch = FunctionalDef(
  inputs  = List("in1", "in2"),
  layers  = Map("d1" -> Dense(32), "d2" -> Dense(32), "out" -> Output(1)),
  connections = List(("in1","d1"), ("in2","d2"), ("d1","out"), ("d2","out")),
  outputs = List("out")
)

// DL4J: Backend.compileGraph(arch) → ComputationGraph
// DJL:  Backend.compile(arch)      → LambdaBlock (v0.5.3)
```

---

## Why Two Backends?

| | DJL | DL4J |
|---|-----|------|
| **Engine** | PyTorch 2.7.1 native | JVM-native (C++ ND4J) |
| **Training** | GPU + CPU | GPU + CPU + Spark distributed |
| **Python deps** | libtorch (auto-downloaded) | None |
| **Best for** | Cloud GPU, PyTorch ecosystem | On-prem JVM, big data pipelines |
| **Maintainer** | AWS (150 contributors) | Konduit (1 maintainer) |
| **Releases** | Monthly (0.36.0) | ~3 years (1.0.0-M2.1) |

---

## Contributing

Add a new backend in 3 files:
1. `Backend.scala` — `ModelDef` → framework Block (~50 lines)
2. `wrappers.scala` — `ZModel` with `predict()`/`fit()` (~80 lines)
3. `exports.scala` — `export` into `zio.nn` package (1 line)

---

## ONNX & Other Engines

`zio-nn-djl` supports any DJL engine — just pass the engine name:

```scala
// PyTorch (default)
ZModel.create(arch, "m", engine = "PyTorch")

// ONNX Runtime — train in PyTorch/TensorFlow, serve on JVM
ZModel.load(Path.of("model.onnx"), engine = "OnnxRuntime")

// TensorFlow
ZModel.load(Path.of("saved_model"), engine = "TensorFlow")

// MXNet, PaddlePaddle, XGBoost, LightGBM
ZModel.load(path, engine = "MXNet")
```

**DJL engines:** PyTorch, OnnxRuntime, TensorFlow, MXNet, PaddlePaddle, XGBoost, LightGBM, fastText, SentencePiece.

No additional module needed — add `onnxruntime` to dependencies if using ONNX models:

```scala
libraryDependencies += "com.microsoft.onnxruntime" % "onnxruntime" % "1.19.2"
```

---

## License

Apache 2.0

---

## ZIO-Native API (v0.3.0)

```scala
import zio.nn.djl.zioApi.*
ZIO.scoped {
  for
    model <- create(arch, "m")           // ZIO[Scope, Throwable, ZModel]
    pred  <- model.predictZ(features)    // Task[Array[Float]]
    // model auto-closed by Scope
  yield pred
}
```

## Tensor Operations (v0.5.0)

ZIO-wrapped tensor math for both backends — identical API, zero code change when swapping:

```scala
import zio.nn.TensorOps.*  // resolves to DL4J or DJL backend automatically
for
  a <- createDouble1D(Array(1.0, 2.0, 3.0))
  b <- createDouble1D(Array(0.5, 0.5, 0.5))
  c <- add(a, b); d <- matMul(a, b); e <- toDoubleArray(c)
yield e
```

Ops: `create`, `create1D`, `createDouble`, `createDouble1D`, `add`, `sub`, `mul`, `div`, `matMul`, `dot`, `transpose`, `sum`, `mean`, `neg`, `toFloatArray`, `toDoubleArray`, `shape`

## Conv2D / CNN (v0.5.3)

```scala
Sequential(1)(  // 1 = input channels
  Conv2D(32, (3,3)), MaxPool2D((2,2)),
  Conv2D(64, (3,3)), MaxPool2D((2,2)),
  Flatten, Dense(128, ReLU), Output(10, Softmax)
).withConvInput(28, 28, 1)  // height, width, channels
 .build
```

`withConvInput` is required when the first layer is Conv2D — it tells
the DL4J backend to configure `InputType.convolutional(28, 28, 1)`
for automatic dimension calculation through the pooling and flatten layers.

---

## TensorOps Guide

See [TENSOROPS.md](TENSOROPS.md) for full usage examples and operation reference.

---

## ZIO Stream Integration (v0.6.0)

Streaming prediction and online training via ZIO Stream:

```scala
import zio.nn.zioApi.*
import zio.stream.*

ZIO.scoped {
  create(arch, "stream-model").flatMap { model =>

    // Streaming prediction — each chunk predicted immediately
    val predictions: ZStream[Any, Throwable, Array[Float]] =
      featureStream.via(model.predictFlow)

    // Streaming training — online SGD from infinite stream
    val losses: ZStream[Any, Throwable, FitResult] =
      dataStream.via(model.fitFlow(epochs = 1, lr = 0.001f))

    // Use case: real-time signals from WebSocket
    priceStream
      .grouped(32)               // mini-batches of 32
      .via(model.predictFlow)    // stream predictions
      .map(preds => if preds.head > threshold) Action.Buy else Action.Hold)
      .runCollect
  }
}
```

| Method | In | Out | Use case |
|--------|-----|-----|----------|
| `predictFlow` | `ZStream[... Array[Array[Float]]]` | `ZStream[... Array[Float]]` | Live prediction |
| `fitFlow` | `ZStream[... (Array[Array[Float]], Array[Float])]` | `ZStream[... FitResult]` | Online training |

---

## Metrics & Checkpointing (v0.7.0)

Timed predictions and training with auto-logging:

```scala
import zio.nn.zioApi.*

// Timed prediction — logs duration at DEBUG level
model.predictTimed(features)

// Timed training — logs duration + loss at INFO level
model.fitTimed(features, labels, epochs = 50)

// Training with periodic checkpoints
model.fitWithCheckpoints(
  features, labels,
  epochs = 100, saveEvery = 10,
  checkpointPath = "models/lstm"
)
// Saves: models/lstm-epoch10, models/lstm-epoch20, ...
```

**Adding Prometheus:** Use `zio-metrics-connectors` + `@@ Metric.timer(...)` for full observability:

```scala
val predictions = model.predictZ(features) @@
  Metric.timer("predict_ms").tagged("model", "lstm-v2")
```

---

## ZIO Config (v0.7.1)

Define model architectures in HOCON/YAML — swap without recompiling:

```hocon
# application.conf
model {
  sequential {
    input-size = 7
    layers = [
      { lstm: { n-in = 7, n-out = 64, activation = tanh } }
      { dense: { n-in = 64, n-out = 32, activation = relu } }
      { output: { n-in = 32, n-out = 1, loss = mse } }
    ]
    optimizer = { adam: { learning-rate = 0.001 } }
    seed = 42
  }
}
```

```scala
// Load from config — zero hardcoded architectures
import zio.nn.ConfigLoader

val arch: Task[ModelDef] = ConfigLoader.fromHocon("model")
arch.flatMap(a => ZModel.create(a, "from-config"))
```

**Benefits:** A/B test architectures without recompilation. Analysts edit HOCON, not Scala.
