# zio-nn User Guide

## Architecture

zio-nn has four layers that work together to let you write framework-agnostic neural network code:

```mermaid
flowchart TD
    A["Your Code"] --> B["DSL Layer (core)"]
    B --> C["Backend Layer (djl / dl4j)"]
    C --> D["Embeddings Module (dl4j-embeddings)"]
    C --> E["Native Framework"]

    A1["Sequential(7)(LSTM(64), Dense(32), Output(1))"]
    A2["ZModel.create(arch, &quot;m&quot;).get"]
    A3["model.predict(features)"]
    A1 --> A2 --> A3

    B1["ModelDef / SequentialDef / LayerDef / ActivationFn"]
    B2["Pure data — zero framework deps"]
    B1 --> B2

    C1["Backend.compile(ModelDef) → Block / MultiLayerNetwork"]
    C2["ZModel wraps native model objects"]
    C1 --> C2

    E1["ai.djl.Model (PyTorch/ONNX/TF)"]
    E2["MultiLayerNetwork (DL4J)"]
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

```mermaid
sequenceDiagram
    participant User as Your Code
    participant ZM as ZModel
    participant TO as TensorOps
    participant Native as Framework

    User->>ZM: predict(features: Array[Array[Float]])
    ZM->>TO: convert to NDList / INDArray
    TO->>Native: model.predict(nativeTensor)
    Native-->>TO: NDList / INDArray output
    TO-->>ZM: convert to Array[Float]
    ZM-->>User: Array[Float]
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

```mermaid
stateDiagram-v2
    [*] --> Created: ZModel.create(arch, name)
    Created --> Trained: fit(features, labels, 50)
    Created --> Saved: save(path)
    Created --> Predict: predict(features)
    Predict --> Predict: predict(features) (reuse)
    Predict --> Trained: fit(...)
    Trained --> Saved: save(path)
    Saved --> Loaded: ZModel.load(path)
    Loaded --> Predict: predict(features)
    Loaded --> Trained: fit(...)
    Predict --> Closed: close()
    Trained --> Closed: close()
    Saved --> Closed: close()
    Loaded --> Closed: close()
    Closed --> [*]
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

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
\tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

GRU:

$$
\begin{aligned}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \\
\tilde{h}_t &= \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

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

### DL4J → ONNX Export

DL4J models can be exported to ONNX protobuf format without any protobuf-java dependency. The ONNX binary is written by hand using the protobuf wire format (`DataOutputStream`).

```scala
import zio.nn.dl4j.Dl4jToOnnx
import java.nio.file.Path

// Export to byte array
val bytes = Dl4jToOnnx.toOnnx(model.underlying)       // Try[Array[Byte]]

// Export directly to file
Dl4jToOnnx.saveToFile(model.underlying, Path.of("model.onnx"))  // Try[Unit]

// Or via the ZModel companion
ZModel.toOnnx(model.underlying, Path.of("model.onnx"))           // Try[Unit]

// ZIO variant
import zio.nn.dl4j.zioApi.*
ZIO.scoped {
  create(arch, "m").flatMap(_.toOnnxZ(Path.of("model.onnx")))   // Task[Unit]
}
```

**Supported layers:** Dense, Output, LSTM, Bidirectional, Embedding, BatchNormalization, Dropout (identity at inference), RnnOutputLayer.

**Not supported:** GRU, MultiHeadAttention, Conv2D, MaxPool2D, Flatten, LayerNorm.

The exported ONNX file can be loaded with DJL's `OnnxRuntime` engine for inference on GPU or edge devices.

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

### Pure Computation Methods (v0.10.0)

`ActivationFn` and `LossFn` now carry pure computation methods — no backend needed:

```scala
import zio.nn.*

// Activation functions
ActivationFn.ReLU.apply(-1.0)                          // 0.0
ActivationFn.Sigmoid.apply(0.0)                         // 0.5
ActivationFn.Tanh.derivative(0.5)                       // matches 1 - tanh²(x)
ActivationFn.Softmax.applyVector(Array(2.0, 1.0, 0.1))  // sums to 1.0

// Loss functions
LossFn.MSE.compute(Array(0.0, 2.0), Array(1.0, 2.0))   // 0.5
LossFn.MAE.compute(Array(1.0, 3.0), Array(4.0, 5.0))   // 2.5
LossFn.BinaryCrossEntropy.compute(Array(0.9), Array(1.0)) // ~0.105
```

These are useful for evaluation, monitoring, and debugging without needing a GPU or backend framework. The issue specification details are in [#27](https://github.com/szekai/zio-nn/issues/27).

### LayerNorm

Layer normalization — normalizes across the feature dimension:

```scala
import zio.nn.dsl.*

val arch = Sequential(512)(
  Dense(512, ReLU),
  LayerNorm,                          // normalizes activations
  Output(10, Softmax)
).build
```

- DL4J: not available (DL4J 1.0.0-M2.1 lacks LayerNorm). Use DJL backend.
- DJL: compiles to native `LayerNorm` block.

### TransformerEncoder

Full transformer encoder block — stacks of self-attention + feed-forward with residual connections:

```scala
import zio.nn.dsl.*

val arch = Sequential(512)(
  TransformerEncoder(dim = 512, numHeads = 8, ffDim = 2048, numLayers = 2, dropout = 0.1),
  Output(10, Softmax)
).build
```

Parameters: `dim` (model dimension), `numHeads` (attention heads), `ffDim` (feed-forward hidden size), `numLayers` (stack count), `dropout`.

- DL4J: not available — use DJL backend or compose manually via `FunctionalDef`.
- DJL: compiles to native `TransformerEncoderBlock` per layer.

## Advanced Training: Callbacks & Early Stopping

ZIO-native training hooks for per-epoch logging, early stopping, and LR scheduling:

```scala
import zio.nn.*

// Log epoch metrics
val logger = new TrainingCallback {
  def onEvent(event: TrainingEvent) = event match
    case TrainingEvent.EpochEnd(epoch, loss, ms) =>
      ZIO.logInfo(s"Epoch $epoch: loss=$loss (${ms}ms)")
    case TrainingEvent.TrainEnd(result) =>
      ZIO.logInfo(s"Done: ${result.epochs} epochs, final loss=${result.loss}")
    case _ => ZIO.unit
}

// Early stopping — stops when validation loss plateaus
val earlyStop = EarlyStopping(patience = 5, minDelta = 0.001)

// Cosine LR scheduling
val cosineLR = LRSchedule.cosine(minLr = 0.0001f, maxLr = 0.01f, cycleLength = 10)

// Combine everything
import zio.nn.dl4j.zioApi.*
ZIO.scoped {
  create(arch).flatMap { model =>
    model.fitWithCallbacksZ(feats, labels, 100,
      callbacks = List(logger, earlyStop),
      lrSchedule = cosineLR,
      validationData = Some((valFeats, valLabels)))
  }
}
```

`fit()` now returns per-epoch loss history:

```scala
val result = model.fit(features, labels, 50).get
result.lossHistory.length  // 50 — one per epoch
result.validationLoss      // Some(0.123) if validation data provided
```

## Evaluation Metrics (v0.10.0)

Built-in classification metrics and `evaluate()` on ZModel:

```scala
import zio.nn.*

// Available metrics
EvalMetric.Accuracy
EvalMetric.Precision(posLabel = 1.0)
EvalMetric.Recall(posLabel = 1.0)
EvalMetric.F1(posLabel = 1.0)

// Direct computation (pure, no model needed)
val pred = Array(0.9, 0.1, 0.8)
val actual = Array(1.0, 0.0, 1.0)
EvalMetric.Accuracy.compute(pred, actual)  // 1.0 (perfect)
EvalMetric.F1().compute(pred, actual)      // 1.0

// Evaluate on a model (both backends)
import zio.nn.dl4j.ZModel
val model = ZModel.create(arch).get
model.evaluate(features, labels, List(EvalMetric.Accuracy, EvalMetric.F1()))
// Returns: Map("accuracy" → 0.87, "f1(pos=1.0)" → 0.85)

// ZIO variant
import zio.nn.dl4j.zioApi.*
ZIO.scoped {
  create(arch).flatMap { model =>
    model.evaluateZ(feats, labels, List(EvalMetric.Accuracy))
  }
}
```

Multi-class support: when the model output dimension > 1, the flat predictions are reshaped to `(samples, classes)` and argmax is applied before computing each metric. The issue specification details are in [#29](https://github.com/szekai/zio-nn/issues/29).

### Population Stability Index (PSI)

Monitor model drift by comparing two probability distributions:

```scala
import zio.nn.EvaluationMetrics.*

val expected = Array(0.2, 0.3, 0.5)    // reference distribution
val actual   = Array(0.1, 0.6, 0.3)    // current distribution

// Default: 10 bins, 1e-10 smoothing
val psi = EvaluationMetrics.psi(expected, actual).get  // Double

// Custom configuration
val psi2 = EvaluationMetrics.psi(expected, actual, numBins = 20, smoothing = 1e-8).get
```

## Batch Data Loading: DataSetLoader

ZIO Stream pipeline from files on disk → transform → batched arrays → model:

```scala
import zio.nn.*, zio.nn.dl4j.zioApi.*
import java.nio.file.Path

// Image classification
val pipeline = ImagePipeline(Resize(28, 28), Normalize(mnistMean, mnistStd))
DataSetLoader.fromImageDir(Path.of("data/train"), pipeline, batchSize = 64) {
  (bytes, pl) => ImageTransformer(pl).transform(bytes)
}.flatMap { loader =>
  loader.batches.via(model.fitFlow(epochs = 1)).runCollect
}

// Text classification with Word2Vec
val tok = w2v.toTokenizer()
DataSetLoader.fromTextDir(Path.of("data/imdb/train"), batchSize = 32) {
  (text, _) => tok.encode(text).map(_.tokenIds)
}.flatMap { loader =>
  loader.batches.via(model.fitFlow).runCollect
}
```

Labels are extracted automatically from parent directory names (`LabelExtractor.fromParentDir`).

## Checkpoint Load/Resume (v0.10.0)

`fitWithCheckpoints` saves periodic model snapshots. The checkpoint utilities let you inspect, resume, and clean them:

```scala
import zio.nn.dl4j.zioApi.*  // or zio.nn.djl.zioApi.* for DJL

ZIO.scoped {
  create(arch).flatMap { model =>
    // Train with checkpoints
    _ <- model.fitWithCheckpoints(feats, labels, 100,
           saveEvery = 10, checkpointPath = "models/lstm")
    // Creates: models/lstm-epoch10, models/lstm-epoch20, ...
  }
}

// List all checkpoints
listCheckpoints("models/lstm")        // Task[List[TrainingCheckpoint]]
// Returns sorted by epoch ascending

// Read a single checkpoint's metadata
loadCheckpoint("models/lstm-epoch10") // Task[TrainingCheckpoint]

// Resume training from the latest checkpoint
ZIO.scoped {
  resumeFromCheckpoint("models/lstm").flatMap { case (loaded, cp) =>
    loaded.fitZ(moreFeats, moreLabels, epochs = 10)
  }
}

// Keep latest N, delete the rest
cleanCheckpoints("models/lstm", keep = 3)  // Task[Unit]
```

The `TrainingCheckpoint` case class carries epoch number, file path, metadata map, and timestamp. The issue specification details are in [#28](https://github.com/szekai/zio-nn/issues/28).

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

## DataSetIterator Training (DL4J) (v0.11.0)

Train and evaluate a DL4J model using a streaming `DataSetIterator` instead of
loading all data into memory as `Array[Array[Float]]`:

```scala
import zio.nn.dl4j.ZModel
import zio.nn.dl4j.zioApi.*
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import scala.jdk.CollectionConverters.*

// Build an iterator from in-memory data
val iterator: DataSetIterator = {
  val features = xorInputs.map(Nd4j.create)     // Array[Array[Float]] → INDArray
  val labels   = xorLabels.map(l => Nd4j.create(Array[Float](l)))
  val datasets = features.zip(labels).map((f, l) => DataSet(f, l))
  ListDataSetIterator(datasets.toList.asJava, 2)
}

ZIO.scoped {
  create(arch).flatMap { model =>
    for
      result  <- model.fitIteratorZ(iterator, epochs = 200, lr = 0.1f)
      // result: FitResult(loss = 0.03, epochs = 200, lossHistory = List(...))

      metrics <- model.evaluateIteratorZ(iterator, List(EvalMetric.Accuracy()))
      // metrics: Map("accuracy" -> 1.0)
    yield ()
  }
}
```

The iterator is `reset()` at the start of each epoch automatically. This API
works with any DL4J `DataSetIterator` — `MnistDataSetIterator`,
`RecordReaderDataSetIterator`, `ListDataSetIterator`, or custom implementations.

See `examples/src/main/scala/zio/nn/examples/Dl4jIteratorExample.scala` for a
complete runnable example.

## Dataset Training (DJL) (v0.11.0)

Train and evaluate a DJL model using a DJL `Dataset`:

```scala
import zio.nn.djl.ZModel
import zio.nn.djl.zioApi.*
import ai.djl.ndarray.{NDArray, NDArrays, NDManager}
import ai.djl.training.dataset.{ArrayDataset, Dataset}

// Build a dataset from in-memory data
def buildDataset(ndm: NDManager): Dataset = {
  val data   = ndm.create(xorInputs.flatten).reshape(4, 2)
  val labels = ndm.create(xorLabels.map(Array[Float](_).toArray).flatten).reshape(4, 1)
  ArrayDataset.Builder()
    .setData(data).optLabels(labels)
    .build()
}

ZIO.scoped {
  ZIO.fromAutoCloseable(ZIO.attempt(NDManager.newBaseManager())).flatMap { implicit ndm =>
    create(arch).flatMap { model =>
      for
        trainDs <- ZIO.attempt(buildDataset(ndm))
        result  <- model.fitDatasetZ(trainDs, epochs = 200, batchSize = 2, lr = 0.1f)
        // result: FitResult(loss = 0.03, epochs = 200)

        evalDs <- ZIO.attempt(buildDataset(ndm))
        metrics <- model.evaluateDatasetZ(evalDs, batchSize = 2, List(EvalMetric.Accuracy()))
        // metrics: Map("accuracy" -> 1.0)
      yield ()
    }
  }
}
```

The DJL `Dataset` abstraction supports both in-memory (`ArrayDataset`) and
random-access (`RandomAccessDataset`) sources, making it suitable for large
datasets that don't fit in memory.

See `examples/src/main/scala/zio/nn/examples/DjlDatasetExample.scala` for a
complete runnable example.

## RNN Time-Step Inference (v0.11.0)

Feed inputs one step at a time while preserving the RNN's internal hidden
state — the building block for auto-regressive generation and real-time
sequence prediction:

```scala
import zio.nn.dl4j.ZModel
import zio.nn.dl4j.zioApi.*
import org.nd4j.linalg.factory.Nd4j

ZIO.scoped {
  create(arch).flatMap { model =>
    for
      // Prime the state with a seed sequence
      _     <- model.rnnTimeStepZ(Nd4j.create(Array(Array(0.1f))))
      _     <- model.rnnTimeStepZ(Nd4j.create(Array(Array(0.2f))))

      // Generate: feed previous output back as next input
      out1  <- model.rnnTimeStepZ(Nd4j.create(Array(Array(0.3f))))   // continues from seed state
      _     <- ZIO.logInfo(s"Step output: $out1")

      // Reset state when starting a new independent sequence
      _     <- model.rnnClearPreviousStateZ
      fresh <- model.rnnTimeStepZ(Array(Array(0.5f)))   // starts from zero state
    yield ()
  }
}
```

- `rnnTimeStepZ` — single time-step forward pass (preserves hidden state)
- `rnnClearPreviousStateZ` — reset hidden state to zero for a new sequence

**Difference from `predictZ`:** `predictZ` treats each call as an independent
forward pass. `rnnTimeStepZ` accumulates state across calls, so step N+1
receives the hidden state from step N. Call `rnnClearPreviousStateZ` to reset.

See `examples/src/main/scala/zio/nn/examples/RnnTimeStepExample.scala` for a
complete runnable example.

## Examples Module

The `examples/` module contains self-contained, runnable example programs that
demonstrate the full lifecycle for each new API:

```bash
sbt "examples/runMain zio.nn.examples.Dl4jIteratorExample"   # DL4J DataSetIterator
sbt "examples/runMain zio.nn.examples.DjlDatasetExample"     # DJL Dataset
sbt "examples/runMain zio.nn.examples.RnnTimeStepExample"    # RNN time-step
```

Just run `sbt "examples/run"` for an interactive menu.

## Vector Database / RAG (v0.10.0)

Store model predictions in a vector database for retrieval-augmented generation (RAG), semantic search, and similarity queries.

### VectorStore Trait

The `VectorStore` trait in `zio.nn` defines a framework-agnostic API for storing and searching float vectors:

| Method | Signature | Description |
|--------|-----------|-------------|
| `store` | `Try[Unit]` | Store a single `VectorRecord(id, values, metadata)` |
| `storeBatch` | `Try[Unit]` | Batch store multiple records |
| `search` | `Try[Seq[VectorRecord]]` | Search by query vector, return k nearest |
| `delete` | `Try[Unit]` | Delete a record by id |
| `deleteBatch` | `Try[Unit]` | Delete multiple records by ids |
| `close` | `Unit` | Release resources |

### InMemoryVectorStore

For testing and local use — stores vectors in a concurrent hash map with cosine distance search:

```scala
import zio.nn.vectordb.InMemoryVectorStore

val store = InMemoryVectorStore()
store.store(VectorRecord("a", Array(1f, 0f, 0f), Map("color" -> "red"))).get
val results = store.search(Array(0.9f, 0.1f, 0.0f), 2).get
```

### PgvectorStore

PostgreSQL + pgvector for production — uses Magnum DB and pgvector JDBC:

```scala
import zio.nn.vectordb.*

// Managed resource with ZIO config
ZIO.scoped {
  createPgvectorStore.flatMap { store =>
    store.storeZ(VectorRecord("q", Array(1f, 0f, 0f))) *>
      store.searchZ(Array(0.9f, 0.1f, 0.0f), 5)
  }
}
```

Requires PostgreSQL with pgvector extension. Configuration via HOCON:

```hocon
vectordb {
  url = "jdbc:postgresql://localhost:5432/vectordb"
  user = "postgres"
  password = "secret"
  table = "vectors"
  dimension = 768
}
```

### ZIO Wrappers

All `VectorStore` methods have ZIO variants via the `vectordb` package object:

```scala
import zio.nn.vectordb.*

store.storeZ(record)           // Task[Unit]
store.storeBatchZ(records)     // Task[Unit]
store.searchZ(query, k)        // Task[Seq[VectorRecord]]
store.deleteZ(id)              // Task[Unit]
store.deleteBatchZ(ids)        // Task[Unit]
```

### Egress Pipeline: predictAndStoreFlow

Combine model prediction with vector storage in a single streaming pipeline. Each stream element is a `(features, ids)` batch — features are predicted, results are stored as `VectorRecord`s and emitted:

```scala
import zio.nn.dl4j.zioApi.*  // or zio.nn.djl.zioApi.*

ZIO.scoped {
  for
    model <- create(arch, "rag-model")
    store <- ZIO.succeed(InMemoryVectorStore())
    _     <- ZStream(
               (Array(Array.fill(7)(0.5f), Array.fill(7)(0.3f)), Array("a", "b")),
               (Array(Array.fill(7)(0.9f)), Array("c"))
             ).via(model.predictAndStoreFlow(store))
              .runCollect
  yield ()
}
```

The `predictAndStore` (Try) and `predictAndStoreZ` (Task) variants are also available for non-streaming use:

```scala
// Try-based
model.predictAndStore(features, store, ids).get

// ZIO
model.predictAndStoreZ(features, store, ids)
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

`vocabRange` controls which Unicode code points are included in the vocabulary (default: `32 to 126` — printable ASCII):

```scala
// Include Latin-1 Supplement characters
val tok = ZTokenizer.regex("\\w+", vocabRange = 32 to 255).get

// Full Unicode range
val tok = ZTokenizer.regex("\\W+", vocabRange = 0 to 0x10FFFF).get
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

## References

```bibtex
@article{hochreiter1997lstm,
  author  = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  year    = {1997},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  doi     = {10.1162/neco.1997.9.8.1735}
}

@article{cho2014gru,
  author  = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  title   = {Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation},
  journal = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year    = {2014},
  pages   = {1724--1734},
  doi     = {10.3115/v1/D14-1179}
}
```
