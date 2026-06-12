## zio-nn core

Framework-agnostic neural network definitions and DSL. No backend dependencies.

## Architecture

- **ModelArchitecture.scala** — Core ADTs: `ModelDef` (enum: Sequential, Functional), `SequentialDef`, `LayerDef` (enum: Dense, LSTM, Output, BatchNorm, Dropout, Conv2D, MaxPool2D, Flatten, Embedding), `AdvancedLayerDef` (enum: GRU, BiDirectional, MultiHeadAttention), `AnyLayer` wrapper, `ActivationFn` (enum: Tanh, ReLU, Sigmoid, Softmax, Identity, LeakyReLU), `LossFn` (enum: MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy, Huber), `OptimizerDef` (sealed: Adam, SGD, RMSprop), `FitResult` (case class: loss, epochs)
- **dsl.scala** — Fluent builder syntax: `Sequential(nInputs)(Layer*)` creates `SequentialDef`, chain `.withOptimizer(Adam(0.001))`, `.withSeed(42L)`, `.build` returns `ModelDef`. Shape-inference: Dense, LSTM, Output, BatchNorm, Dropout, Conv2D, MaxPool2D, Flatten, Embedding, GRU, BiDirectional, MultiHeadAttention. Defaults: Dense → ReLU, Output → MSE+Identity, LSTM → Tanh. Conv2D support via `.withConvInput(h, w, ch)`.
- **ConfigLoader.scala** — ZIO Config integration. `fromHocon("path")` parses HOCON/YAML `ModelDef` config using raw Typesafe Config (case-insensitive enum parsing). Externalizes architecture definition — no recompilation needed for model changes.
- **Preprocessing.scala** — Framework-agnostic preprocessing types: `TokenizerConfig` (padding, addSpecialTokens), `EncodingResult` (tokenIds, attentionMask, tokenTypeIds), `ImageTransformDef` (enum: Resize, Normalize, CenterCrop), `ImagePipeline` (seq of ImageTransformDef). Zero backend dependencies — backend implementation is in djl/ and dl4j/.

## Source Files

| File | Path | Description |
|------|------|-------------|
| ModelArchitecture.scala | `core/src/main/scala/zio/nn/ModelArchitecture.scala` | Core ADTs: ModelDef, LayerDef, ActivationFn, LossFn, OptimizerDef, FitResult |
| dsl.scala | `core/src/main/scala/zio/nn/dsl.scala` | Fluent builder: Sequential, Dense, LSTM, Output, BatchNorm, Dropout |
| ConfigLoader.scala | `core/src/main/scala/zio/nn/ConfigLoader.scala` | ZIO Config integration for HOCON/YAML model definitions |
| Preprocessing.scala | `core/src/main/scala/zio/nn/Preprocessing.scala` | TokenizerConfig, EncodingResult, ImageTransformDef, ImagePipeline |

## Key Rules

- **Zero framework imports** in this module. No DJL, DL4J, or ND4J types allowed.
- **No runtime** — pure type definitions + DSL builder logic. No training, no inference.
- **Pattern match on ModelDef** in backends to dispatch architecture.
- **DSL defaults** must match documented behavior in README.md.

## Test Patterns

- File: `DSLSpec.scala` (12 tests), `ModelArchitectureSpec.scala` (10 tests)
- Tests pattern-match on return values to verify structure (e.g., `arch match { case ModelDef.Sequential(s) => ... }`)
- All build assertions use `assertTrue()`.
- No resource management needed — core types are pure data.

## Boundary Rules

- Never add backend-specific code here.
- Never produce runtime side effects.
- Never import `ai.djl.*` or `org.deeplearning4j.*`.
