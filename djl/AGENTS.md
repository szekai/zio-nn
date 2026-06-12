## zio-nn djl

Deep Java Library (DJL) backend — PyTorch/ONNX/TF/MXNet engine support.

## Architecture

- **Backend.scala** — `Backend.compile(arch: ModelDef)` → DJL `Block`. Pattern-matches on `ModelDef.Sequential` / `FunctionalDef`, constructs `SequentialBlock` with layers. Supports: Dense, LSTM, GRU, BiDirectional, MultiHeadAttention, Conv2D, MaxPool2D, Flatten, Embedding (via IdEmbedding), BatchNorm. Dropout is identity. `compileGraph(arch)` for multi-input DAGs → `LambdaBlock`.
- **wrappers.scala** — `ZModel` wrapping DJL `ZooModel`. Methods: `create`, `predict`, `fit`, `save`, `load`, `close`. Unified API contract: `predict(Array[Array[Float]])` → `Try[Array[Float]]`.
- **implicits.scala** — `Array[Float]` ↔ `NDArray` via `.toNDArray`/`.toFloatArray`, `Array[Array[Float]]` ↔ `NDList` via `.toNDList`/`.toFloatArrays`. Requires implicit `NDManager`.
- **scope.scala** — `withNDManager(program: ZIO[Any, Throwable, T])` manages NDManager lifecycle. Sub-manager per call for safety. Optional — internal base manager handles default resource lifecycle.
- **tensor/** — `TensorOps` for NDArray math: `add`, `sub`, `mul`, `div`, `matMul`, `dot`, `transpose`, `sum`, `mean`, `neg`, `create`, `create1D`, `createDouble`, `createDouble1D`. All ops wrapped in ZIO.
- **zio.scala** — ZIO-native wrappers: `predictZ`, `predictDoubleZ`, `fitZ`, `predictFlow`, `fitFlow` (ZStream), `predictTimed`, `fitTimed`, `fitWithCheckpoints`, plus tokenizer wrappers (`huggingFaceTokenizer`, `encodeZ`, `batchEncodeZ`, `decodeZ`).
- **ZTokenizer.scala** — HuggingFace tokenizer wrapper: `ZTokenizer.huggingFace(modelName)`, `fromJson(path)`. Methods: `encode`, `batchEncode`, `decode`. Requires `"ai.djl.huggingface" % "tokenizers"` on classpath.
- **ImageTransformer.scala** — DJL CV transform pipeline: `ImageTransformer(pipeline)` applies `Resize`, `Normalize`, `CenterCrop` via NDImageUtils. `transform(bytes)` → `Try[Array[Array[Float]]]`. Requires implicit NDManager.

## Key Rules

- Always wrap `NDManager` resource lifecycle — use `scope.withNDManager` or internal base manager.
- `Dropout` is identity in DJL (not supported during inference by PyTorch engine).
- Always provide implicit `NDManager` in scope for implicits usage.
- Export all public symbols into `zio.nn` package via package object.

## Test Patterns

- Files: `BackendSpec.scala` (5), `ImplicitsSpec.scala` (1), `ZIOApiSpec.scala` (3), `TensorOpsSpec.scala` (4), `ZTokenizerSpec.scala` (2 + 5 ignored), `ImageTransformerSpec.scala` (6)
- Backend tests verify `Backend.compile(arch) != null` for all layer types + activation combos.
- Implicits test does `Array[Float] → NDArray → Array[Float]` roundtrip.
- Need explicit `NDManager` (via `newBaseManager()`) in implicits tests.
- `ZTokenizerSpec` has 5 ignored tests (need HuggingFace model download — requires network).
- `ImageTransformerSpec` construction tests don't need NDManager; transform tests use `provideLayer(ndmLayer)`.
- DJL uses `Test / fork := true` because PyTorch native library can only be loaded once per JVM.

## DJL-Specific Notes

- Engine auto-detected from classpath (PyTorch default). Override via `model.engine = "OnnxRuntime"`.
- `predictorRaw()` / `trainerRaw()` escape hatches for custom training loops.
- DJL releases monthly (~0.36.x). Active maintenance by AWS.
- GPU requires libtorch shared library (auto-downloaded by DJL).
- **HuggingFace tokenizers**: Add `"ai.djl.huggingface" % "tokenizers" % djlV` to dependencies for `ZTokenizer`.
- **Test isolation**: Always use `Test / fork := true` for DJL module tests to avoid native library classloader conflicts.
