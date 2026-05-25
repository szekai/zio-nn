[← szekai.github.io](https://szekai.github.io/)

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

📖 **[User Guide →](USER_GUIDE.md)** — architecture, DSL reference, escape hatches, backend swap guide.

## Modules

| Module | Description |
|--------|-------------|
| `zio-nn-core` | DSL (`dsl.*`), types (`ModelDef`, `LayerDef`, `FitResult`), ConfigLoader, implicits |
| `zio-nn-djl` | ZModel, Backend, zioApi, TensorOps, scope, implicits — PyTorch/ONNX/TF/XGBoost |
| `zio-nn-dl4j` | ZModel, Backend, zioApi, TensorOps, implicits — JVM-native (no Python) |
| `zio-nn-dl4j-embeddings` | Word2Vec training (`SequenceVectors + SkipGram`), pre-trained vector loading, embedding-to-LayerSpec bridge |

```scala
// sbt
libraryDependencies += "io.github.szekai" %% "zio-nn-djl" % "0.7.2"  // or zio-nn-dl4j
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
  case Success(result) => println(s"Loss: ${result.loss}")
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
| `Embedding(10000, 300)` | `LayerDef.Embedding(vocabSize=10000, embeddingDim=300)` |
| `Conv2D(32, (3,3))` | `LayerDef.Conv2D(nIn=auto, filters=32, kernel=(3,3), ReLU)` |
| `MaxPool2D((2,2))` | `LayerDef.MaxPool2D(poolSize=(2,2))` |
| `Flatten` | `LayerDef.Flatten` |
| `GRU(64, Tanh)` | `AdvancedLayerDef.GRU(nIn=auto, nOut=64, Tanh)` (DJL only) |
| `BiDirectional(LSTM(64))` | `AdvancedLayerDef.BiDirectional(LSTM, nIn=auto, nOut=64, Tanh)` |
| `MultiHeadAttention(300, 8)` | `AdvancedLayerDef.MultiHeadAttention(embeddingDim=300, numHeads=8)` |

Input sizes auto-propagate through the chain: `Sequential(7)(LSTM(64), Dense(32), Output(1))` — the compiler resolves 7→64, 64→32, 32→1 automatically.

For embedding models, `Sequential(1)` starts with token-index input and the `Embedding` layer self-declares `vocabSize`/`embeddingDim`:

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
| Embedding | ✅ | ✅ |
| GRU | ✅ | ✅ |
| BiDirectional (LSTM/GRU) | ✅ | ✅ |
| MultiHeadAttention | ✅ | ✅ |

### GRU vs LSTM

GRU (Gated Recurrent Unit) is a simpler alternative to LSTM with fewer gates and no separate cell state:

**LSTM equations** (3 gates + cell state):
```
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)   // forget gate
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)   // input gate
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)   // output gate
c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c) // cell candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t     // cell state
h_t = o_t ⊙ tanh(c_t)                // hidden state
```

**GRU equations** (2 gates, no cell state):
```
z_t = σ(W_z·[h_{t-1}, x_t] + b_z)   // update gate
r_t = σ(W_r·[h_{t-1}, x_t] + b_r)   // reset gate
h̃_t = tanh(W_h·[r_t ⊙ h_{t-1}, x_t] + b_h)
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

GRU has fewer parameters and often converges faster. Use GRU when you want faster training; use LSTM when you need the extra expressiveness of a separate cell state.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Your Code (import zio.nn.dsl.*, zio.nn.*)              │
│  Sequential(1)(Embedding(10k,300), BiDirectional(       │
│    LSTM(256)), MultiHeadAttention(256,8), Output(2))    │
├─────────────────────────────────────────────────────────┤
│  DSL Layer (core)                                       │
│  LayerDef (LSTM,Dense,Output,BatchNorm,Dropout,Conv2D,  │
│    MaxPool2D,Flatten,Embedding)                         │
│  + AdvancedLayerDef (GRU,BiDirectional,MultiHeadAttn)   │
│  Wrapped in AnyLayer for unified SequentialDef          │
├─────────────────────────────────────────────────────────┤
│  Backend Layer (djl / dl4j)                             │
│  Backend.compile(ModelDef) → Block / MultiLayerNetwork  │
│  ZModel wraps native model objects                      │
├─────────────────────────────────────────────────────────┤
│  Embeddings Module (dl4j-embeddings)                    │
│  Word2Vec.train(), .load(), .similarity(), .wordsNearest│
│  Word2VecModel → EmbeddingWeights → LayerSpec bridge    │
├─────────────────────────────────────────────────────────┤
│  Native Framework                                       │
│  ai.djl.Model (PyTorch/ONNX/TF) | MultiLayerNetwork     │
└─────────────────────────────────────────────────────────┘
```

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

---

## Word2Vec Embeddings (v0.7.2)

Load pre-trained vectors and use them as the first layer in your model:

```scala
import zio.nn.*, zio.nn.dsl.*
import zio.nn.dl4j.embeddings.*

// Load Google News Word2Vec
val w2v = Word2Vec.loadGoogleNewsVectors(Path.of("GoogleNews-vectors-negative300.bin")).get

// Use as first layer with pre-trained weights
val arch = Sequential(1)(
  w2v.toEmbeddingLayer(),    // vocabSize + dim + weights auto-detected
  LSTM(256, Tanh),
  Output(2, Softmax)
).build

val model = ZModel.create(arch, "imdb-sentiment").get

// Predict with token-index input
model.predictInt(Array(Array(42)))      // Try[Array[Float]]
model.fitInt(tokens, labels, epochs=5)  // Try[FitResult]

// Word2Vec similarity queries
val sim  = w2v.similarity("day", "night")      // Task[Double]
val near = w2v.wordsNearest("king", 10)         // Task[List[String]]

// GloVe vectors (escape hatch)
val glove = Word2Vec.loadGloVe(Path.of("glove.6B.300d.txt")).get
// Convert to EmbeddingWeights manually, then use Embedding(vocabSize, dim, weights)
```

| Method | Backend | Description |
|--------|---------|-------------|
| `Embedding(vocabSize, dim)` | DJL + DL4J | Randomly initialized embedding layer |
| `Word2Vec.load(path)` | DL4J only | Load pre-trained SequenceVectors |
| `Word2Vec.loadGloVe(path)` | DL4J only | Load GloVe .txt vectors (escape hatch) |
| `Word2Vec.loadGoogleNewsVectors(path)` | DL4J only | Load Google News .bin vectors |
| `toEmbeddingLayer()` | DL4J only | Convert vectors → LayerSpec.Embedding with weights |
| `similarity(w1, w2)` | DL4J only | Cosine similarity between two words |
| `wordsNearest(w, n)` | DL4J only | Top-N most similar words |
| `predictInt(tokens)` | DJL + DL4J | Predict from token-index input |
| `fitInt(tokens, labels, epochs)` | DJL + DL4J | Train from token-index input |

**DJL Note**: Both backends support embeddings natively. DJL uses `IdEmbedding` for integer-index embeddings.

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
