## zio-nn djl

Deep Java Library (DJL) backend — PyTorch/ONNX/TF/MXNet engine support.

## Architecture

- **Backend.scala** — `Backend.compile(arch: ModelDef)` → DJL `Block`. Pattern-matches on `ModelDef.Sequential` / `FunctionalDef`, constructs `SequentialBlock` with layers. Supports: Dense, LSTM, GRU, BiDirectional, MultiHeadAttention, Conv2D, MaxPool2D, Flatten, Embedding (via IdEmbedding), BatchNorm. Dropout is identity. `compileGraph(arch)` for multi-input DAGs → `LambdaBlock`.
- **wrappers.scala** — `ZModel` wrapping DJL `ZooModel`. Methods: `create`, `predict`, `fit`, `save`, `load`, `close`. Unified API contract: `predict(Array[Array[Float]])` → `Try[Array[Float]]`.
- **implicits.scala** — `Array[Float]` ↔ `NDArray` via `.toNDArray`/`.toFloatArray`, `Array[Array[Float]]` ↔ `NDList` via `.toNDList`/`.toFloatArrays`. Requires implicit `NDManager`.
- **scope.scala** — `withNDManager(program: ZIO[Any, Throwable, T])` manages NDManager lifecycle. Sub-manager per call for safety. Optional — internal base manager handles default resource lifecycle.
- **tensor/** — `TensorOps` for NDArray math: `add`, `sub`, `mul`, `div`, `matMul`, `dot`, `transpose`, `sum`, `mean`, `neg`, `create`, `create1D`, `createDouble`, `createDouble1D`. All ops wrapped in ZIO.
- **zio.scala** — ZIO-native wrappers: `predictZ`, `predictDoubleZ`, `fitZ`, `predictFlow`, `fitFlow` (ZStream), `predictTimed`, `fitTimed`, `fitWithCheckpoints`.

## Key Rules

- Always wrap `NDManager` resource lifecycle — use `scope.withNDManager` or internal base manager.
- `Dropout` is identity in DJL (not supported during inference by PyTorch engine).
- Always provide implicit `NDManager` in scope for implicits usage.
- Export all public symbols into `zio.nn` package via package object.

## Test Patterns

- File: `BackendSpec.scala` (5 tests), `ImplicitsSpec.scala` (1 test)
- Backend tests verify `Backend.compile(arch) != null` for all layer types + activation combos.
- Implicits test does `Array[Float] → NDArray → Array[Float]` roundtrip.
- Need explicit `NDManager` (via `newBaseManager()`) in implicits tests.
- DJL has fewer integration tests than DL4J (no full fit→predict roundtrip yet).

## DJL-Specific Notes

- Engine auto-detected from classpath (PyTorch default). Override via `model.engine = "OnnxRuntime"`.
- `predictorRaw()` / `trainerRaw()` escape hatches for custom training loops.
- DJL releases monthly (~0.36.x). Active maintenance by AWS.
- GPU requires libtorch shared library (auto-downloaded by DJL).
