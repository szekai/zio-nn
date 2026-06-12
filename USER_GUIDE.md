# zio-nn User Guide

## Architecture

zio-nn has four layers that work together to let you write framework-agnostic neural network code:

```
┌─────────────────────────────────────────────────────────┐
│  Your Code                                              │
│  import zio.nn.dsl.*                                     │
│  import zio.nn.dl4j.ZModel  // or zio.nn.djl.ZModel      │
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
│  Embeddings Module (dl4j-embeddings)                    │
│  Word2Vec.train(), .load(), .toEmbeddingLayer()         │
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
import zio.nn.dl4j.ZModel   // or zio.nn.djl.ZModel for DJL

// DL4J
val model = ZModel.create(arch)
// Under the hood: Backend.compile(arch) → ZModel.wrap(net)

// DJL
import zio.nn.djl.ZModel
val djlModel = ZModel.create(arch, "my-model").get
// Under the hood: Backend.compile(arch) → Model.newInstance() → ZModel
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

### 5. Backend-specific imports (since v0.9.0)

Each backend module defines its types (`ZModel`, `Backend`) in its own package:

- `zio.nn.djl.ZModel` and `zio.nn.djl.Backend` for the DJL backend
- `zio.nn.dl4j.ZModel` and `zio.nn.dl4j.Backend` for the DL4J backend

`import zio.nn.*` provides only the framework-agnostic core types (`ModelDef`, `FitResult`, `ActivationFn`, etc.) from `zio-nn-core`. Backend types require explicit imports:

```scala
import zio.nn.*              // core types only (ModelDef, FitResult, ...)
import zio.nn.dl4j.ZModel    // backend-specific
import zio.nn.dl4j.Backend
```

**Why this matters**: Earlier versions used `exports.scala` to re-export backend types into `zio.nn.*`, but this caused `NoSuchMethodError` at runtime when both `zio-nn-djl` and `zio-nn-dl4j` were on the classpath. The fix removes the re-exports and requires explicit backend imports — a small ergonomic cost for correct classpath coexistence.

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
import zio.nn.dl4j.zioApi.*  // or zio.nn.djl.zioApi.* for DJL
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
| Embedding | `Embedding(10000, 300)` | Token index → dense vector (first layer only) |
| GRU | `GRU(64, Tanh)` | Gated recurrent unit (2 gates, no cell state) |
| BiDirectional | `BiDirectional(LSTM(64))` | Bidirectional wrapper, doubles output dim |
| MultiHeadAttention | `MultiHeadAttention(300, 8)` | Multi-head self-attention (Transformer) |

### GRU vs LSTM Equations

LSTM (6 equations, 3 gates + cell state) vs GRU (4 equations, 2 gates):

```
LSTM:  f_t = σ(W_f·[h_{t-1},x_t]+b_f)  i_t = σ(W_i·[h_{t-1},x_t]+b_i)
       o_t = σ(W_o·[h_{t-1},x_t]+b_o)  c̃_t = tanh(W_c·[h_{t-1},x_t]+b_c)
       c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t  h_t = o_t ⊙ tanh(c_t)

GRU:   z_t = σ(W_z·[h_{t-1},x_t]+b_z)  r_t = σ(W_r·[h_{t-1},x_t]+b_r)
       h̃_t = tanh(W_h·[r_t ⊙ h_{t-1},x_t]+b_h)
       h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### Word2Vec Embeddings (DL4J only)

```scala
import zio.nn.dl4j.embeddings.*

// ── Load pre-trained vectors ──
val w2v = Word2Vec.loadGoogleNewsVectors(Path.of("vectors.bin")).get

// Use as first layer with pre-trained weights
val arch = Sequential(1)(
  w2v.toEmbeddingLayer(),    // vocabSize + dim + weights auto-detected
  LSTM(256, Tanh),
  Output(2, Softmax)
).build

val model = ZModel.create(arch).get

// Predict / train with integer token indices
model.predictInt(Array(Array(42)))       // Try[Array[Float]]
model.fitInt(tokens, labels, epochs = 5) // Try[FitResult]

// Word2Vec queries
w2v.similarity("day", "night")           // Task[Double]
w2v.wordsNearest("king", 10)             // Task[List[String]]

// ── Train from scratch ──
val corpus = ZStream("the cat sat on the mat", "the dog sat on the log")
ZIO.scoped {
  Word2Vec.train(corpus, Config(dimensions = 50, epochs = 3)).flatMap { w2v =>
    w2v.similarity("cat", "mat")
  }
}

// ── Tokenizer from Word2Vec vocabulary ──
val tok = w2v.toTokenizer()
val ids = tok.encode("hello world").get.tokenIds
model.fitInt(Array(ids), labels, epochs = 5)

// ── Escape hatch for legacy DL4J pipelines ──
val raw: WordVectors = w2v.vectors
val iter = new DataSetIteratorWord2Vec(dataDir, raw, batch, trunc, train)
```

DJL users: export PyTorch `nn.Embedding` as ONNX, load via `ZModel.load(path, engine="OnnxRuntime")`.

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
import zio.nn.dl4j.zioApi.*  // or zio.nn.djl.zioApi.* for DJL
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

## Tokenization

Convert text to token IDs for models with `Embedding` as the first layer.

### DJL — HuggingFace Tokenizers

```scala
import zio.nn.djl.ZTokenizer

// Auto-download from HuggingFace hub
val tok = ZTokenizer.huggingFace("bert-base-uncased").get
val result = tok.encode("hello world").get
// result.tokenIds: Array[Int]
// result.attentionMask: Option[Array[Int]]
// result.tokenTypeIds: Option[Array[Int]]

// Decode back
val decoded = tok.decode(result.tokenIds).get
tok.close()
```

### DL4J — Regex / Whitespace Tokenizers

```scala
import zio.nn.dl4j.ZTokenizer

// Regex tokenizer — split on non-word characters
val tok = ZTokenizer.regex("\\W+").get
val result = tok.encode("a b c").get     // 3 tokens
tok.close()

// Whitespace tokenizer — split on spaces
val tok2 = ZTokenizer.whitespace()
tok2.encode("hello world").get.tokenIds // 2 tokens
tok2.close()
```

### ZIO Wrappers

```scala
import zio.nn.dl4j.zioApi.*  // or zio.nn.djl.zioApi.* for DJL

ZIO.scoped {
  for
    tok     <- huggingFaceTokenizer("bert-base-uncased")  // auto-closed
    encoded <- tok.encodeZ("hello world")
    decoded <- tok.decodeZ(encoded.tokenIds)
  yield decoded
}
```

## Image Preprocessing

Transform raw image bytes into float arrays for vision model inputs.

```scala
import zio.nn.*
import zio.nn.dl4j.ImageTransformer    // or zio.nn.djl.ImageTransformer

// Build a transformation pipeline
val pipeline = ImagePipeline(
  ImageTransformDef.Resize(224, 224),
  ImageTransformDef.Normalize(
    mean = Array(0.485f, 0.456f, 0.406f),
    std  = Array(0.229f, 0.224f, 0.225f)
  ),
  ImageTransformDef.CenterCrop(200, 200)
)

// Apply to raw image bytes
val transformer = ImageTransformer(pipeline)
val pixels: Try[Array[Array[Float]]] =
  transformer.transform(Files.readAllBytes(Path.of("image.jpg")))
```

Available transforms: `Resize(height, width)`, `Normalize(mean, std)`, `CenterCrop(height, width)`.

## Tensor Operations

## Implicit Conversions

Bridge between unified `Array[Float]` and native tensor types:

```scala
import zio.nn.djl.implicits.*   // DJL backend
val nd: NDList = features.toNDList      // unified → native
val back: Array[Array[Float]] = nd.toFloatArrays  // native → unified

import zio.nn.dl4j.implicits.*  // DL4J backend
val ind: INDArray = features.toINDArray  // unified → native
```

## Backend Swap

Change one line in `build.sbt` — zero code changes:

```scala
"io.github.szekai" %% "zio-nn-djl"  % "0.8.0"  // ← swap
"io.github.szekai" %% "zio-nn-dl4j" % "0.8.0"  // ← zero code changes
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
