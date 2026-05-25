## zio-nn dl4j

DeepLearning4j (DL4J) backend — JVM-native training with optional Spark distributed support.

## Architecture

- **Backend.scala** — `Backend.compile(arch: ModelDef)` → DL4J `MultiLayerNetwork`. Pattern-matches `ModelDef.Sequential`, configures `NeuralNetConfiguration.List` builder with layers. Supports: Dense, LSTM, GRU, BiDirectional, MultiHeadAttention, Conv2D, MaxPool2D, Flatten, Embedding (via EmbeddingSequenceLayer), BatchNormalization, Dropout. `compileGraph(arch)` for multi-input DAGs → `ComputationGraph`.
- **wrappers.scala** — `ZModel` wrapping DL4J `MultiLayerNetwork`. Methods: `create`, `predict`, `fit`, `save`, `load`, `close`. Unified API contract: `predict(Array[Array[Float]])` → `Try[Array[Float]]`.
- **implicits.scala** — `Array[Float]` ↔ `INDArray` via `.toINDArray`/`.toFloatArray`, `Array[Array[Float]]` ↔ `INDArray` via `.toINDArray`/`.toFloatArrays`. No implicit manager needed (ND4J handles memory).
- **tensor/** — `TensorOps` for INDArray math. Same operations as DJL TensorOps.
- **zio.scala** — ZIO-native wrappers: `predictZ`, `predictDoubleZ`, `fitZ`, `predictFlow`, `fitFlow` (ZStream), `predictTimed`, `fitTimed`, `fitWithCheckpoints`.

## Key Rules

- No NDManager or scope equivalent — ND4J has JVM-managed memory (GC + `close()`).
- `Dropout` works correctly in DL4J (unlike DJL).
- Always close `MultiLayerNetwork` via `model.close()` when done.
- `ComputationGraph` for multi-input DAGs (FunctionalDef) — use `Backend.compileGraph`.
- Export all public symbols into `zio.nn` package via package object.

## Test Patterns

- File: `BackendSpec.scala` (6 tests), `ImplicitsSpec.scala` (2 tests), `WrappersSpec.scala` (4 tests), `EdgeCaseSpec.scala` (7 tests), `ZIOApiSpec.scala` (3 tests)
- Most comprehensive test suite — 22 tests total.
- Covers: compile, predict, fit, save/load roundtrip, zero epochs, single sample, idempotent close, ZIO Scope management.
- `EdgeCaseSpec` is the integration test template: inline random data, temp files with `.deleteOnExit()`, manual `.close()`.
- `WrappersSpec` verifies unified API signatures (`ZModel.create`, `.predict`, `.fit`).
- `ZIOApiSpec` verifies `.provideLayer(Scope.default)` resource management.

## DL4J-Specific Notes

- DL4J 1.0.0-M2.1 is final — no new releases from Konduit. Bugfixes are community-only.
- Spark distributed training via `SparkDl4jMultiLayer` (not in unified API — use `model.underlying`).
- ND4J memory: JVM-GC managed, but `.close()` on INDArray can free off-heap memory early.
- GPU support requires nd4j-cuda-* backend on classpath.
