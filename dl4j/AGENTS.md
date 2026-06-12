## zio-nn dl4j

DeepLearning4j (DL4J) backend — JVM-native training with optional Spark distributed support.

## Architecture

- **Backend.scala** — `Backend.compile(arch: ModelDef)` → DL4J `MultiLayerNetwork`. Pattern-matches `ModelDef.Sequential`, configures `NeuralNetConfiguration.List` builder with layers. Supports: Dense, LSTM, GRU, BiDirectional, MultiHeadAttention, Conv2D, MaxPool2D, Flatten, Embedding (via EmbeddingSequenceLayer), BatchNormalization, Dropout. `compileGraph(arch)` for multi-input DAGs → `ComputationGraph`.
- **wrappers.scala** — `ZModel` wrapping DL4J `MultiLayerNetwork`. Methods: `create`, `predict`, `fit`, `save`, `load`, `close`. Unified API contract: `predict(Array[Array[Float]])` → `Try[Array[Float]]`.
- **implicits.scala** — `Array[Float]` ↔ `INDArray` via `.toINDArray`/`.toFloatArray`, `Array[Array[Float]]` ↔ `INDArray` via `.toINDArray`/`.toFloatArrays`. No implicit manager needed (ND4J handles memory).
- **tensor/** — `TensorOps` for INDArray math. Same operations as DJL TensorOps.
- **zio.scala** — ZIO-native wrappers: `predictZ`, `predictDoubleZ`, `fitZ`, `predictFlow`, `fitFlow` (ZStream), `predictTimed`, `fitTimed`, `fitWithCheckpoints`, plus tokenizer wrappers (`regexTokenizer`, `whitespaceTokenizer`, `encodeZ`, `batchEncodeZ`, `decodeZ`).
- **ZTokenizer.scala** — Regex/whitespace tokenizer: `ZTokenizer.regex(pattern)`, `ZTokenizer.whitespace()`. Methods: `encode`, `batchEncode`, `decode`. No external dependencies — local tokenization only.
- **ImageTransformer.scala** — ND4J image transform pipeline: `ImageTransformer(pipeline)` applies `Resize`, `Normalize`, `CenterCrop` via Java2D + ND4J. `transform(bytes)` → `Try[Array[Array[Float]]]`. No implicit manager needed.

## Key Rules

- No NDManager or scope equivalent — ND4J has JVM-managed memory (GC + `close()`).
- `Dropout` works correctly in DL4J (unlike DJL).
- Always close `MultiLayerNetwork` via `model.close()` when done.
- `ComputationGraph` for multi-input DAGs (FunctionalDef) — use `Backend.compileGraph`.
- Export all public symbols into `zio.nn` package via package object.

## Test Patterns

- Files: `BackendSpec.scala` (6), `ImplicitsSpec.scala` (2), `WrappersSpec.scala` (4), `EdgeCaseSpec.scala` (7), `ZIOApiSpec.scala` (3), `ZTokenizerSpec.scala` (4), `ImageTransformerSpec.scala` (3)
- Most comprehensive test suite — 37 tests total.
- Covers: compile, predict, fit, save/load roundtrip, zero epochs, single sample, idempotent close, ZIO Scope management, regex/whitespace tokenization, image transform pipeline.
- `EdgeCaseSpec` is the integration test template: inline random data, temp files with `.deleteOnExit()`, manual `.close()`.
- `WrappersSpec` verifies unified API signatures (`ZModel.create`, `.predict`, `.fit`).
- `ZIOApiSpec` verifies `.provideLayer(Scope.default)` resource management.
- `ZTokenizerSpec` tests regex and whitespace tokenizers with encode/batchEncode/decode roundtrip.
- `ImageTransformerSpec` tests Resize/Normalize/CenterCrop pipeline on synthetic PNG images.

## DL4J-Specific Notes

- DL4J 1.0.0-M2.1 is final — no new releases from Konduit. Bugfixes are community-only.
- Spark distributed training via `SparkDl4jMultiLayer` (not in unified API — use `model.underlying`).
- ND4J memory: JVM-GC managed, but `.close()` on INDArray can free off-heap memory early.
- GPU support requires nd4j-cuda-* backend on classpath.
