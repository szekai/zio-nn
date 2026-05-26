# Plan: Word2Vec Embedding Support in zio-nn

## 1. Goal

Add embedding layer support to zio-nn so that:
1. Pre-trained embeddings (Google News Word2Vec, custom SequenceVectors) can be used as the **first layer** in zio-nn sequential models (GloVe via escape hatch — see Step 6a)
2. Word2Vec Skip-Gram training (`SequenceVectors + SkipGram`) gets a ZIO-native API via a new `zio-nn-dl4j-embeddings` module
3. The zlearning project can replace raw DL4J code with zio-nn DSL + embedding module

### Target usage after implementation

```scala
// ── Path A: Pre-trained embeddings as first layer ──
import zio.nn.*, zio.nn.dsl.*
import zio.nn.dl4j.embeddings.*

val w2v = Word2Vec.load(Path.of("GoogleNews-vectors-negative300.bin")).get

val arch = Sequential(1)(
  w2v.toEmbeddingLayer(),           // vocabSize + dim + weights auto-detected
  LSTM(256, Tanh),
  Output(2, Softmax)
).build

val model = ZModel.create(arch, "imdb-lstm").get
// EmbeddingLayer is initialized with pre-trained weights. Feed tokenized
// sentences as Array[Array[Int]] via the new predictInt() / fitInt() methods.

// ── Path B: Randomly initialized embedding (learned during training) ──
val arch2 = Sequential(1)(
  Embedding(vocabSize = 10000, embeddingDim = 300),   // fresh, no pre-training
  LSTM(256, Tanh),
  Output(2, Softmax)
).build

// ── Path C: Train Word2Vec from scratch (new module) ──
import zio.nn.dl4j.embeddings.*

ZIO.scoped {
  for
    w2v   <- Word2Vec.train(corpus, Config(dimensions = 300, algorithm = Algorithm.SkipGram, epochs = 3))
    sim   <- w2v.similarity("day", "night")       // Task[Double]
    near  <- w2v.wordsNearest("king", 10)          // Task[List[String]]
    _     <- w2v.save(Path.of("models/vectors"))
  yield ()
}
```

---

## 2. Background: Word2Vec vs EmbeddingLayer

### Two distinct use cases

| | EmbeddingLayer (neural net) | Word2Vec training (algorithm) |
|---|---|---|
| **What** | A layer that maps token IDs → dense vectors | A self-supervised algorithm that learns vectors from co-occurrence |
| **Input** | Integer token indices | Raw text corpus |
| **Output** | Continuous vector per token | Trained embedding table |
| **Training** | Backpropagation through the network | Skip-Gram / CBOW with negative sampling |
| **DL4J API** | `EmbeddingSequenceLayer.Builder()` (outputs 3D for RNN) | `SequenceVectors + SkipGram()` |
| **DJL API** | `Embedding.builder()` | ❌ No equivalent (use PyTorch/Gensim) |
| **Serialization** | Embedding weights in model file | Standalone `.bin`/`.txt` file |

**This plan covers both paths.**

---

## 3. Design Decisions

### Decision 1: `EmbeddingWeights` — framework-free core type

Pre-trained vectors need to flow from the embeddings module (DL4J-dependent) into core `LayerDef` types (framework-free). A plain-data bridge type lives in core:

```scala
case class EmbeddingWeights(vocabulary: Map[String, Int], vectors: Array[Array[Float]])
// vocabulary: word → token ID
// vectors:     vectors(tokenId) → float array of length embeddingDim
```

This lets `Word2VecModel` in the embeddings module produce `EmbeddingWeights` by extracting vocabulary + vector table from DL4J's `SequenceVectors`, then embed them in a `LayerDef.Embedding`. The backends read `pretrained` and initialize the embedding layer accordingly — no framework leak into core.

### Decision 2: `LayerDef.Embedding` carries optional pre-trained weights

```scala
case Embedding(vocabSize: Int, embeddingDim: Int, pretrained: Option[EmbeddingWeights] = None)
```

- `None` → randomly initialized (standard neural network usage)
- `Some(weights)` → initialized from pre-trained vectors (transfer learning path)

This avoids a separate "injection" method. The backend reads the field and initializes accordingly.

### Decision 3: New `zio-nn-dl4j-embeddings` module (training path)

Word2Vec Skip-Gram training is fundamentally different from layer-based training:
- It's self-supervised (no labels)
- Uses different loss (negative sampling)
- Different DL4J API surface (`SequenceVectors`, not `MultiLayerNetwork`)

Trying to force it into the `ModelDef`/`ZModel` type system would be awkward. Instead, a separate module wraps DL4J's `SequenceVectors` with ZIO effects — similar to how `zio-nn-dl4j` wraps `MultiLayerNetwork`.

The module also provides the bridge: `Word2VecModel.toEmbeddingWeights()` → `EmbeddingWeights` and `Word2VecModel.toEmbeddingLayer()` → `LayerSpec` (for use in DSL sequential builder).

### Decision 4: DJL backend — escape hatch only

DJL's `ai.djl.nn.core.Embedding<T>` block requires a `Collection<T>` of items via `.setItems()`, not a simple integer count. For integer-index embeddings, pass `(0 until vocabSize).map(Integer.valueOf).asJavaCollection`. DJL's Embedding builder has NO method for pre-trained weight tables — pre-trained weights must be injected post-build via DJL's `ParameterStore` or loaded via ONNX.

- **Layer path**: DJL compiles `LayerDef.Embedding` → `Embedding<Integer>.builder().setItems(...).setEmbeddingSize(...).build()`. Random initialization only; pre-trained weights are an escape-hatch concern.
- **Training path**: Throw `NotImplementedError` in the embeddings module, with documentation pointing to PyTorch/Gensim for producing ONNX embedding models loadable via DJL.

### Decision 5: Tokenization stays outside zio-nn

zio-nn deals with tensors (arrays of floats/integers). Text → token IDs conversion remains the user's responsibility — same pattern as DL4J's `DataSetIterator`.

### Decision 6: Integer input for embedding-first models

When the first layer is `Embedding`, model input is `Array[Array[Int]]` (batch of tokenized sequences). New methods on `ZModel`:

```scala
def predictInt(tokens: Array[Array[Int]]): Try[Array[Float]]
def predictIntZ(tokens: Array[Array[Int]]): Task[Array[Float]]
```

Each backend converts `Array[Array[Int]]` to its native integer tensor (ND4J `INDArray` with int type / DJL `NDArray` with int type) before forward pass.

### Decision 7: DSL shape propagation for Embedding

For `Embedding(vocabSize, embeddingDim)`, the `resolve(nIn)` method ignores the propagated `nIn` (it's always `1` for token-index input) and produces `LayerDef.Embedding(vocabSize, embeddingDim, ...)`. The `outputSize` returns `embeddingDim`, so downstream layers (LSTM, Dense, etc.) receive the correct dimension.

```scala
// LayerSpec.Embedding:
def resolve(nIn: Int): LayerDef = LayerDef.Embedding(vocabSize, embeddingDim, pretrained)  // nIn ignored
def outputSize: Int = embeddingDim  // downstream layers receive this dimension
```

---

## 4. Files to Modify

### Phase 1 — Core DSL (zio-nn-core)

| File | Change |
|---|---|
| `core/src/main/scala/zio/nn/ModelArchitecture.scala` | Add `EmbeddingWeights` case class + `Embedding(vocabSize, embeddingDim, pretrained?)` to `LayerDef` enum |
| `core/src/main/scala/zio/nn/dsl.scala` | Add `LayerSpec.Embedding` variant, DSL constructor `Embedding(vocabSize, embeddingDim)`, shape propagation |
| `core/src/test/scala/zio/nn/ModelArchitectureSpec.scala` | Tests: Embedding in sequential def, shape propagation, EmbeddingWeights equality |
| `core/src/test/scala/zio/nn/DSLSpec.scala` | Tests: `Sequential(1)(Embedding(10000, 300), Dense(300, 128)).build` resolves correctly |

### Phase 2 — DL4J Backend

| File | Change |
|---|---|
| `dl4j/src/main/scala/zio/nn/dl4j/Backend.scala` | Add `Embedding` → `EmbeddingSequenceLayer.Builder()` in `toDL4JLayer` (NOT plain `EmbeddingLayer` — must output 3D for LSTM downstream); handle `pretrained` weight init |
| `dl4j/src/main/scala/zio/nn/dl4j/wrappers.scala` | Add `predictInt()` / `predictIntZ()` + `fitInt()` / `fitIntZ()` for integer token-index input |
| `dl4j/src/test/scala/zio/nn/dl4j/BackendSpec.scala` | Compile sequential + embedding model, verify layer count + weight init |

### Phase 3 — DJL Backend

| File | Change |
|---|---|
| `djl/src/main/scala/zio/nn/djl/Backend.scala` | Add `Embedding` → `ai.djl.nn.core.Embedding<Integer>.builder()` in `toDJLBlock` (random-init only; pre-trained weights via ONNX escape hatch) |
| `djl/src/main/scala/zio/nn/djl/wrappers.scala` | Add `predictInt()` / `predictIntZ()` + `fitInt()` / `fitIntZ()` for integer token-index input |
| `djl/src/test/scala/zio/nn/djl/BackendSpec.scala` | Compile and verify |

### Phase 4 — Embeddings Module (new)

| File | Change |
|---|---|
| `build.sbt` (root) | Add `lazy val embeddings` subproject definition; register in `.aggregate()` |
| `embeddings/src/main/scala/zio/nn/dl4j/embeddings/Word2Vec.scala` | `Word2Vec.train()`, `.load()`, `.save()`, `.similarity()`, `.wordsNearest()`, `.toEmbeddingWeights()`, `.toEmbeddingLayer()` |
| `embeddings/src/test/scala/zio/nn/dl4j/embeddings/Word2VecSpec.scala` | Train on small corpus, verify nearest neighbors |

**CRITICAL**: Do NOT create `embeddings/build.sbt`. Define `lazy val embeddings` directly in the root `build.sbt` (this repo uses a single root `build.sbt` with all subproject definitions).

### Phase 5 — Documentation

| File | Change |
|---|---|
| `README.md` | Add Embedding to layer table, add Word2Vec example |
| `USER_GUIDE.md` | Add "Embedding Layer" section, add "Word2Vec Training" section |
| `FEATURE_MATRIX.md` (new) | Mark Embedding + Word2Vec coverage per backend |

---

## 5. Implementation Steps

### Step 1: Add `EmbeddingWeights` + `Embedding` LayerDef to core

**File**: `core/src/main/scala/zio/nn/ModelArchitecture.scala`

Add after the `Flatten` case (line 35):

```scala
/** Pre-trained embedding vectors — framework-free bridge type.
  *
  * @param vocabulary word → token ID mapping
  * @param vectors    vectors(tokenId) → float array of length embeddingDim
  */
case class EmbeddingWeights(vocabulary: Map[String, Int], vectors: Array[Array[Float]])
```

Add to `LayerDef` enum (after `Flatten`):
```scala
/** Maps discrete token IDs to dense vector representations.
  * When used as first layer, model input is `Array[Array[Int]]` (token indices).
  * Set `pretrained = Some(EmbeddingWeights(...))` to initialize from pre-trained vectors.
  *
  * @param vocabSize     number of unique tokens in vocabulary
  * @param embeddingDim  dimension of each embedding vector (e.g. 300 for Google News)
  * @param pretrained    pre-trained weights to initialize the embedding table (None = random init)
  */
case Embedding(vocabSize: Int, embeddingDim: Int, pretrained: Option[EmbeddingWeights] = None)
```

### Step 2: Add DSL constructor + LayerSpec

**File**: `core/src/main/scala/zio/nn/dsl.scala`

Add to `LayerSpec` enum:
```scala
case Embedding(vocabSize: Int, embeddingDim: Int, pretrained: Option[EmbeddingWeights] = None)

def resolve(nIn: Int): LayerDef = this match
  ...
  case LayerSpec.Embedding(vocabSize, embeddingDim, pretrained) =>
    LayerDef.Embedding(vocabSize, embeddingDim, pretrained)  // nIn ignored for Embedding

def outputSize: Int = this match
  ...
  case LayerSpec.Embedding(_, embeddingDim, _) => embeddingDim
```

Add DSL constructor (after `Flatten`, before `// Activation shortcuts`):
```scala
/** Create a randomly-initialized embedding layer.
  *
  * @param vocabSize    number of unique tokens
  * @param embeddingDim output vector dimension
  */
def Embedding(vocabSize: Int, embeddingDim: Int): LayerSpec =
  LayerSpec.Embedding(vocabSize, embeddingDim, None)

/** Create an embedding layer with pre-trained weights.
  * Called by `Word2VecModel.toEmbeddingLayer()` in the embeddings module.
  */
def Embedding(vocabSize: Int, embeddingDim: Int, weights: EmbeddingWeights): LayerSpec =
  LayerSpec.Embedding(vocabSize, embeddingDim, Some(weights))
```

**Shape propagation logic**: When `Sequential(1)(Embedding(10000, 300), Dense(300, 128), Output(128, 2))` is built:
- Start: nIn = 1
- Embedding: `resolve(1)` → `LayerDef.Embedding(10000, 300, None)`, `outputSize` = 300, next nIn = 300
- Dense: `resolve(300)` → `LayerDef.Dense(300, 128, ReLU)`, next nIn = 128
- Output: `resolve(128)` → `LayerDef.Output(128, 2, MSE)`, done

The `nIn` parameter is ignored for Embedding because the layer declares its own `vocabSize` and `embeddingDim`.

### Step 3: DL4J Backend — compile Embedding as sequence layer

**File**: `dl4j/src/main/scala/zio/nn/dl4j/Backend.scala`

**CRITICAL**: Use `EmbeddingSequenceLayer`, not `EmbeddingLayer`. `EmbeddingSequenceLayer` outputs 3D `[batch, sequenceLength, embeddingDim]` which LSTM accepts directly. A plain `EmbeddingLayer` outputs 2D and would require an `InputPreProcessor` (e.g., `FeedForwardToRnnPreProcessor`) to connect to recurrent layers — avoid this complexity.

Add import at line 10:
```scala
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer
```

Add case in `toDL4JLayer` (after `MaxPool2D` case):
```scala
case LayerDef.Embedding(vocabSize, embeddingDim, pretrained) =>
  val builder = new EmbeddingSequenceLayer.Builder().nIn(vocabSize).nOut(embeddingDim)
  pretrained match
    case Some(_) =>
      // Pre-trained weight injection happens in wrappers.scala post-compile
      // via model.getLayer(0).setParam("W", weightTable).
      builder.weightInit(WeightInit.ZERO).build()
    case None =>
      builder.weightInit(WeightInit.XAVIER).build()
```

**File**: `dl4j/src/main/scala/zio/nn/dl4j/wrappers.scala`

Add to `ZModel.create` companion object method (or a new overload):

```scala
/** Create a model and inject pre-trained embedding weights if present. */
def createWithEmbedding(arch: ModelDef): ZModel =
  val model = ZModel(Backend.compile(arch))
  // Find embedding layers and inject pre-trained weights
  arch match
    case ModelDef.Sequential(s) =>
      s.layers.zipWithIndex.foreach {
        case (LayerDef.Embedding(vocabSize, embeddingDim, Some(EmbeddingWeights(_, vectors))), idx) =>
          val wTable = Nd4j.create(vectors.map(_.map(_.toDouble)))
          model.underlying.getLayer(0).setParam("W", wTable)
          model.underlying.getLayer(0).setParam("b", Nd4j.zeros(embeddingDim.toLong))
        case _ => ()
      }
    case _ => ()
  model
```

Add `predictInt` and `fitInt` methods to `ZModel`:
```scala
/** Predict with integer token indices (when first layer is Embedding). */
def predictInt(tokens: Array[Array[Int]]): Try[Array[Float]] =
  Try {
    val indArray = Nd4j.create(tokens.map(_.map(_.toFloat)))
    model.output(indArray).data().asFloat()
  }

/** Train with integer token indices (when first layer is Embedding). */
def fitInt(tokens: Array[Array[Int]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[FitResult] =
  Try {
    val ds = new DataSet(Nd4j.create(tokens.map(_.map(_.toFloat))), Nd4j.create(labels.map(Array(_))))
    for _ <- 1 to epochs do model.fit(ds)
    FitResult(model.score(ds), epochs)
  }
```

### Step 4: DJL Backend — compile Embedding block

**File**: `djl/src/main/scala/zio/nn/djl/Backend.scala`

Add import:
```scala
import ai.djl.nn.core.Embedding as DJLEmbedding
import scala.jdk.CollectionConverters.*
```

Add case in `toDJLBlock` (after `Flatten` case):
```scala
case LayerDef.Embedding(vocabSize, embeddingDim, _) =>
  // DJL Embedding requires actual items, not just a count.
  // Pass integer IDs 0..vocabSize-1 as the item collection.
  val items = (0 until vocabSize).map(Integer.valueOf).asJavaCollection
  DJLEmbedding.builder[Integer]()
    .setItems(items)
    .setEmbeddingSize(embeddingDim)
    .build()
```

> **Note on pre-trained weights for DJL**: `Embedding.Builder` has no `optWeightTable()` or similar method. Pre-trained weight injection requires the DJL escape hatch — load an ONNX model with baked-in weights via `ZModel.load(path, engine="OnnxRuntime")`, or set weights manually via `ParameterStore` on the built block before training. This is documented as an escape-hatch pattern.

**File**: `djl/src/main/scala/zio/nn/djl/wrappers.scala`

Add `predictInt` and `fitInt` methods to `ZModel`:
```scala
/** Predict with integer token indices (when first layer is Embedding). */
def predictInt(tokens: Array[Array[Int]]): Try[Array[Float]] =
  val sub = ndm.newSubManager()
  try
    val flatTokens = tokens.flatten.map(_.toLong)
    val input = new NDList(sub.create(flatTokens, new Shape(tokens.length.toLong, tokens.head.length.toLong)))
    val pred = Predictor(underlying, new NoopTranslator(), Device.cpu(), false)
    try
      val result = pred.predict(input)
      val arr = new Array[Float](result.head().size().toInt)
      System.arraycopy(result.head().toFloatArray, 0, arr, 0, arr.length)
      Try(arr)
    finally pred.close()
  finally sub.close()

/** Train with integer token indices (when first layer is Embedding). */
def fitInt(tokens: Array[Array[Int]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Try[FitResult] =
  Try {
    val adam = Adam.builder().optLearningRateTracker(Tracker.fixed(lr)).build()
    val config = new DefaultTrainingConfig(Loss.l2Loss())
    config.optOptimizer(adam)
    config.optInitializer(new XavierInitializer(), "weight")
    val trainer = underlying.newTrainer(config)
    try
      val flatTokens = tokens.flatten.map(_.toLong)
      trainer.initialize(new Shape(tokens.length.toLong, tokens.head.length.toLong))
      for _ <- 1 to epochs do
        val dataArr = ndm.create(flatTokens, new Shape(tokens.length.toLong, tokens.head.length.toLong))
        val labelArr = ndm.create(labels.map(_.toFloat))
        val batch = new ai.djl.training.dataset.Batch(ndm.newSubManager(), new NDList(dataArr), new NDList(labelArr),
          tokens.length, null, null, tokens.length.toLong, 0L, java.util.Collections.emptyList[Any]())
        ai.djl.training.EasyTrain.trainBatch(trainer, batch); batch.close()
      FitResult(trainer.getTrainingResult.getTrainLoss.toDouble, epochs)
    finally trainer.close()
  }
```

### Step 5: Define embeddings subproject (root build.sbt only)

**File**: `build.sbt` (root)

**Do NOT create `embeddings/build.sbt`** — this repo uses a single root `build.sbt` where all `lazy val` subproject definitions live.

Add the subproject definition after the `dl4j` block:

```scala
// ── DL4J Embeddings: Word2Vec training + pre-trained vector loading ──
lazy val embeddings = project
  .in(file("embeddings"))
  .dependsOn(core, dl4j)
  .settings(
    name := "zio-nn-dl4j-embeddings",
    libraryDependencies ++= Seq(
      "dev.zio"             %% "zio"                    % zioV,
      "dev.zio"             %% "zio-streams"            % zioV,
      "dev.zio"             %% "zio-test"               % zioV % Test,
      "dev.zio"             %% "zio-test-sbt"           % zioV % Test
    )
  )
```

Update the root aggregate:
```scala
lazy val root = project
  .in(file("."))
  .aggregate(core, djl, dl4j, embeddings)
  .settings(
    name := "zio-nn",
    publish / skip := true
  )
```

### Step 6: Implement Word2Vec wrapper

**File**: `embeddings/src/main/scala/zio/nn/dl4j/embeddings/Word2Vec.scala` (new)

```scala
package zio.nn.dl4j.embeddings

import zio.*
import zio.stream.*
import zio.nn.*
import zio.nn.dsl.*
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.word2vec.VocabWord
import java.nio.file.Path
import scala.jdk.CollectionConverters.*

case class Word2VecModel(underlying: SequenceVectors[VocabWord]):

  /** Vocabulary size (number of unique tokens). */
  def vocabSize: Int = underlying.getVocab.numWords()

  /** Embedding dimension. */
  def dim: Int = underlying.getLayerSize

  def similarity(word1: String, word2: String): Task[Double] =
    ZIO.attempt(underlying.similarity(word1, word2))

  def wordsNearest(word: String, n: Int): Task[List[String]] =
    ZIO.attempt(underlying.wordsNearest(word, n).asScala.toList)

  def save(path: Path): Task[Unit] =
    ZIO.attemptBlocking(WordVectorSerializer.writeSequenceVectors(underlying, path.toFile))

  /** Extract vocabulary + vectors as framework-free [[EmbeddingWeights]]. */
  def toEmbeddingWeights: Try[EmbeddingWeights] = Try {
    val words = (0 until vocabSize).map(i => underlying.getVocab.wordFor(i)).toArray
    val vocabMap = words.zipWithIndex.toMap
    val vectors = (0 until vocabSize).map { i =>
      underlying.getLookupTable.getWeights.getRow(i.toLong).dup().data().asFloat()
    }.toArray
    EmbeddingWeights(vocabMap, vectors)
  }

  /** Produce a [[LayerSpec.Embedding]] with pre-trained weights — use in DSL builder.
    * Synchronous extraction (matches zio-nn's Try-first convention).
    */
  def toEmbeddingLayer(): LayerSpec =
    val weights = toEmbeddingWeights.get
    zio.nn.dsl.Embedding(vocabSize, dim, weights)

object Word2Vec:

  enum Algorithm:
    case SkipGram, CBOW

  case class Config(
    dimensions: Int = 300,
    minWordFrequency: Int = 5,
    windowSize: Int = 5,
    batchSize: Int = 512,
    epochs: Int = 3,
    algorithm: Algorithm = Algorithm.SkipGram
  )

  /** Train Word2Vec from a ZStream of sentences (already tokenized, space-separated). */
  def train(corpus: ZStream[Any, Throwable, String], config: Config): ZIO[Scope, Throwable, Word2VecModel] =
    ZIO.attemptBlockingInterrupt {
      import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
      import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator
      import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer
      import org.deeplearning4j.models.word2vec.VocabWord
      import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor
      import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
      import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram
      import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW
      import org.deeplearning4j.models.sequencevectors.SequenceVectors
      import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
      import org.deeplearning4j.models.word2vec.VectorsConfiguration
      import java.nio.file.Files

      // 1. Collect corpus to temp file (DL4J requires file-based input)
      val tmpFile = Files.createTempFile("w2v-corpus-", ".txt")
      tmpFile.toFile.deleteOnExit()
      val succeeded = new java.util.concurrent.atomic.AtomicBoolean(false)
      try
        // Write corpus
        // ... (implementation uses ZStream.runForeach to write lines)
        succeeded.set(true)
        ???
      finally if !succeeded.get() then Files.deleteIfExists(tmpFile)
    }

  /** Load pre-trained vectors (Google News .bin, GloVe .txt, etc.).
    * Returns `Try` — consistent with zio-nn's Try-first convention (`ZModel.create` returns Try).
    *
    * DL4J's `loadStaticModel` returns either `WordVectors` or `SequenceVectors`.
    * We need `SequenceVectors[VocabWord]` for `.getVocab()` and `.getLookupTable()`.
    * For Google News .bin files, `loadStaticModel` returns `SequenceVectors` directly.
    * For GloVe .txt files, it returns `WordVectors` (parent type) — use `loadGoogleNewsVectors` instead.
    */
  def load(path: Path): Try[Word2VecModel] = Try {
    val vecs = WordVectorSerializer.loadStaticModel(path.toFile)
    val seq = vecs match
      case sv: SequenceVectors[?] => sv.asInstanceOf[SequenceVectors[VocabWord]]
      case wv: org.deeplearning4j.models.embeddings.wordvectors.WordVectors =>
        throw new UnsupportedOperationException(
          s"Loaded model is ${wv.getClass.getSimpleName}, not SequenceVectors. " +
          s"Try Word2Vec.loadGoogleNewsVectors() for .bin files, or convert GloVe to SequenceVectors format.")
    Word2VecModel(seq)
  }
```

### Step 6a: GloVe loading (escape hatch)

GloVe `.txt` files are loaded by DL4J as `WordVectors`, not `SequenceVectors[VocabWord]`. The `SequenceVectors`-based `Word2VecModel` wrapper doesn't apply — use the escape hatch directly:

```scala
/** Load GloVe text-format vectors (.txt files from https://nlp.stanford.edu/projects/glove/).
  * Returns raw WordVectors — not wrapped in Word2VecModel. Use `.getWordVectorMatrix()`
  * to extract weights, then construct EmbeddingWeights manually.
  *
  * Example:
  *   val glove = Word2Vec.loadGloVe(Path.of("glove.6B.300d.txt")).get
  *   val table = glove.getWordVectorMatrix  // INDArray
  *   // Convert to EmbeddingWeights manually, then use Embedding(vocabSize, dim, weights)
  */
def loadGloVe(path: Path): Try[org.deeplearning4j.models.embeddings.wordvectors.WordVectors] = Try {
  WordVectorSerializer.loadTxtVectors(path.toFile)
}
```

GloVe vectors can then be converted to `EmbeddingWeights` manually by iterating over the vocabulary and extracting rows from the weight matrix. A convenience method `fromWordVectors(wv: WordVectors): EmbeddingWeights` can be added later.

### Step 6b: Google News vectors (.bin format)

```scala
/** Load Google News Word2Vec .bin file specifically.
  * `readWord2VecModel()` returns `WordVectorsImpl` which IS a `SequenceVectors[VocabWord]`.
  */
def loadGoogleNewsVectors(path: Path): Try[Word2VecModel] = Try {
  val vecs = WordVectorSerializer.readWord2VecModel(path.toFile)
  Word2VecModel(vecs)
}
```

---

## 6. Testing Strategy

### Phase 1 — Core (zio-nn-core)

```bash
# Command: sbt "core/test"
sbt "core/testOnly zio.nn.ModelArchitectureSpec"
```

Expected:
- `Embedding(10000, 300)` in a SequentialDef builds without error
- Shape propagation: `Sequential(1)(Embedding(10000, 300), Dense(300, 128))` → Dense gets nIn=300
- `EmbeddingWeights(Map("a" → 0, "b" → 1), Array(Array(0.1f), Array(0.2f)))` round-trips via equality
- FunctionalDef with Embedding layer compiles graph

```bash
sbt "core/testOnly zio.nn.DSLSpec"
```

Expected:
- `Sequential(1)(Embedding(10000, 300), Dense(300, 128), Output(128, 2)).build` → ModelDef.Sequential with 3 layers

### Phase 2 — DL4J Backend (zio-nn-dl4j)

```bash
# Command: sbt "dl4j/test"
sbt "dl4j/testOnly zio.nn.dl4j.BackendSpec"
```

Expected:
- `Backend.compile(Sequential(1)(Embedding(10000, 300)))` creates MultiLayerNetwork with 1 layer
- Layer 0 is `EmbeddingSequenceLayer` with nIn=10000, nOut=300
- Model with EmbeddingSequenceLayer + LSTM compiles to 2-layer network without preprocessor errors
- `predictInt(Array(Array(42)))` on randomly-initialized embedding model returns float array
- `fitInt(tokenizedInputs, labels, epochs=1)` trains without shape-mismatch errors
- Pre-trained EmbeddingWeights are correctly set on layer 0 after `createWithEmbedding()`

### Phase 3 — DJL Backend (zio-nn-djl)

```bash
# Command: sbt "djl/test"
sbt "djl/testOnly zio.nn.djl.BackendSpec"
```

Expected:
- `Backend.compile(Sequential(1)(Embedding(10000, 300)))` creates non-null Block
- Block has 1 child (the `Embedding<Integer>` block)
- `predictInt(Array(Array(1)))` on model returns float array
- `fitInt(Array(Array(1)), Array(0.5f), epochs=1)` completes without error
- Note: DJL pre-trained weights are not injected via unified API — use ONNX escape hatch

### Phase 4 — Embeddings Module (zio-nn-dl4j-embeddings)

```bash
# Command: sbt "embeddings/test"
sbt "embeddings/testOnly zio.nn.dl4j.embeddings.Word2VecSpec"
```

Expected:
- `Word2Vec.train(ZStream.fromIterable(smallCorpus), Config(...))` returns Word2VecModel
- `Word2Vec.load(Path.of("test-model.bin"))` returns Word2VecModel for valid .bin file
- `Word2Vec.loadGloVe(Path.of("glove.txt"))` returns WordVectors for valid .txt file (escape hatch)
- `model.similarity("day", "night")` returns value in [-1, 1]
- `model.similarity("day", "night")` returns value in [-1, 1]
- `model.wordsNearest("day", 5)` returns list of 5 strings
- `model.toEmbeddingWeights` produces EmbeddingWeights with vocabSize × dim table
- `Word2Vec.save(path)` + `Word2Vec.load(path)` round-trips successfully

### Phase 5 — Documentation

```bash
# Manual verification:
# 1. Check README.md has "Embedding" in the layer table
# 2. Check USER_GUIDE.md has "Embedding Layer" and "Word2Vec Training" sections
# 3. Check FEATURE_MATRIX.md has Embedding + Word2Vec rows per backend
grep -n "Embedding" README.md USER_GUIDE.md
grep -n "Word2Vec" README.md USER_GUIDE.md
```

Expected: non-empty results showing the new sections exist.

### Full verification

```bash
# Run all tests across all modules
sbt test
```

Expected: all tests pass (core, dl4j, djl, embeddings).

---

## 7. Migration Guide: zlearning → zio-nn

### Before (current DL4J raw code — 117 lines)

```scala
// NaturalLanguageModelingDL.scala
val vocabCache   = new AbstractCache.Builder[VocabWord]().build()
val iterator     = new BasicLineIterator(file)
val transformer  = SentenceTransformer.Builder()...
val seqIterator  = AbstractSequenceIterator.Builder[VocabWord](transformer).build()
val constructor  = VocabConstructor.Builder[VocabWord]()...
constructor.buildJointVocabulary(false, true)
val lookupTable  = InMemoryLookupTable.Builder[VocabWord]()
  .vectorLength(300).useAdaGrad(false).cache(vocabCache).build()
val vectors = SequenceVectors.Builder[VocabWord](VectorsConfiguration())
  .minWordFrequency(2).lookupTable(lookupTable).iterate(seqIterator)
  .vocabCache(vocabCache).batchSize(512).epochs(3)
  .elementsLearningAlgorithm(SkipGram[VocabWord]()).build()
vectors.fit()
```

### After (zio-nn embeddings module)

```scala
import zio.nn.dl4j.embeddings.*

val corpus = ZStream.fromFile(Path.of("raw_textual_data"))
  .via(ZPipeline.utf8Decode)
  .via(ZPipeline.splitLines)

ZIO.scoped {
  for
    w2v   <- Word2Vec.train(corpus, Config(dimensions = 300, epochs = 3))
    sim   <- w2v.similarity("day", "night")
    near  <- w2v.wordsNearest("day", 10)
    _     <- ZIO.debug(s"Similarity: $sim, Nearest: $near")
  yield ()
}
```

### Before (IMDB with raw DL4J + manual wrappers)

```scala
val wordVectors = WordVectorSerializer.loadStaticModel(new File(GOOGLE_NEWS_VECTOR_PATH))
val net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
  .list()
  .layer(0, new LSTM.Builder().nIn(300).nOut(256).activation(Activation.TANH).build())
  .layer(1, new RnnOutputLayer.Builder()...).build())
net.init()
```

### After (zio-nn DSL + embedding module)

```scala
import zio.nn.*, zio.nn.dsl.*
import zio.nn.dl4j.embeddings.*

val w2v = Word2Vec.load(Path.of(GOOGLE_NEWS_VECTOR_PATH)).get
val arch = Sequential(1)(w2v.toEmbeddingLayer(), LSTM(256, Tanh), Output(2, Softmax)).build
val model = ZModel.create(arch, "imdb-sentiment").get
// Feed tokenized sentences via model.predictInt(tokenizedReviews)
model.close()
```

---

## 8. Implementation Order & Estimates

| Phase | What | Effort | Depends on | QA Command |
|---|---|---|---|---|
| 1 | `EmbeddingWeights` + `Embedding` in `LayerDef` + DSL | 45 min | — | `sbt "core/test"` |
| 2 | DL4J backend compilation + `predictInt` | 45 min | Phase 1 | `sbt "dl4j/test"` |
| 3 | DJL backend compilation + `predictInt` | 45 min | Phase 1 | `sbt "djl/test"` |
| 4 | `zio-nn-dl4j-embeddings` module | 2-3 hours | Phase 2 | `sbt "embeddings/test"` |
| 5 | Tests (all phases) | 2 hours | Phase 1-4 | `sbt test` |
| 6 | Docs | 1 hour | All | Manual grep + visual check |

**Total**: ~1-2 days

---

## 9. Appendix: Type Flow Diagram

```
┌───────────── core (framework-free) ─────────────┐
│                                                  │
│  EmbeddingWeights                                │
│    ├─ vocabulary: Map[String, Int]               │
│    └─ vectors: Array[Array[Float]]               │
│                                                  │
│  LayerDef.Embedding(vocabSize, dim, pretrained?) │
│  LayerSpec.Embedding(vocabSize, dim, pretrained?)│
│  DSL: Embedding(vocabSize, dim)                  │
│                                                  │
└──────────────┬───────────────┬───────────────────┘
               │               │
     ┌─────────▼──────┐  ┌────▼──────────────┐
     │  DL4J Backend  │  │  DJL Backend      │
     │                │  │                   │
     │ EmbeddingSeq   │  │ Embedding<Integer>│
     │ Layer.Builder  │  │ .setItems(0..N)   │
     │ .nIn(vocab)    │  │ .setEmbedding(dim)│
     │ .nOut(dim)     │  │                   │
     │                │  │ pretrained weights │
     │ if pretrained: │  │ via ONNX escape    │
     │  setParam("W") │  │ hatch only         │
     └────────▲───────┘  └───────────────────┘
              │
  ┌───────────┴─────────────────────┐
  │  zio-nn-dl4j-embeddings module │
  │  (depends on core + dl4j)      │
  │                                 │
  │  Word2VecModel                  │
  │    .train(corpus, Config)       │
  │    .load(path)                  │
  │    .toEmbeddingWeights()        │
  │      → EmbeddingWeights         │
  │    .toEmbeddingLayer()          │
  │      → LayerSpec.Embedding(...) │
  └─────────────────────────────────┘
```

Key: `EmbeddingWeights` is the bridge — plain arrays/floats, no framework types. The embeddings module converts DL4J's `SequenceVectors` → `EmbeddingWeights`, backends convert `EmbeddingWeights` → native weight tables.
