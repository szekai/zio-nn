## zio-nn

Write-once neural network library for ZIO. Swap DJL ↔ DL4J by changing one JAR — zero code changes.

## Scoring Matrix

| Directory | Scala Sources | Tests | Score | AGENTS.md |
|-----------|--------------|-------|-------|-----------|
| root      | (config/build) | — | HIGH | ✅ (this file) |
| core/     | 3 (ModelArchitecture, dsl, ConfigLoader) | 2 files, 22 tests | HIGH | ✅ |
| djl/      | 6 (Backend, implicits, scope, tensor/, wrappers, zio) | 2 files, 6 tests | HIGH | ✅ |
| dl4j/     | 5 (Backend, implicits, tensor/, wrappers, zio) | 5 files, 22 tests | HIGH | ✅ |
| embeddings/ | 1 (Word2Vec) | 0 files, 0 tests | MEDIUM | ❌ |
| macros/   | 0 | 0 | LOW | ❌ |
| project/  | 0 (SBT config only) | 0 | LOW | ❌ |

## Architecture Overview

- **core/** — Framework-agnostic: `ModelDef`, `LayerDef`, `ActivationFn`, `LossFn`, `OptimizerDef`, DSL (`dsl.*`), `ConfigLoader`
- **djl/** — Deep Java Library backend: `ZModel` wrapping `ZooModel`, `Backend.compile` → DJL `Block`, `TensorOps` for NDArray math, `scope.withNDManager` for resource mgmt
- **dl4j/** — DeepLearning4j backend: `ZModel` wrapping `MultiLayerNetwork`, `Backend.compile` → DL4J `MultiLayerNetwork`, `TensorOps` for INDArray math

All backends export into `zio.nn` via package objects. User code imports `zio.nn.*` and `zio.nn.dsl.*`.

## Key Design Decisions

- **No typeclass pattern** — Backend routing is manual (compile against the right package), not typeclass-driven. Intentional: avoids Scala 3 implicit complexity.
- **ZModel is not a trait** — Separate wrapper per backend (no shared interface). Both expose `predict`, `fit`, `save`, `load`, `close` with identical signatures.
- **Try-based API** — Primary API returns `Try[T]`, not `Task[T]`. ZIO wrappers (`predictZ`, `fitZ`, `predictDoubleZ`) live in each backend's `zio.scala`.
- **ZIO Stream integration** — `predictFlow` / `fitFlow` in `zio.scala` per backend.
- **scope.withNDManager** — DJL-only. Optional for NDManager resource lifecycle. Not needed for DL4J.
- **ConfigLoader** — Reads models from HOCON/YAML via ZIO Config. No recompilation needed for architecture changes.

## Guiding Principles

### 1. ZIO-First API Design
Every feature ships with both `Try` (primary, synchronous) and ZIO variants:
- `predict` / `predictZ` — `Try[Array[Float]]` / `Task[Array[Float]]`
- `fit` / `fitZ` — `Try[FitResult]` / `Task[FitResult]`
- `predictFlow` / `fitFlow` — ZIO Stream integration
- `predictTimed` / `fitTimed` — metrics-annotated variants
- `predictInt` / `predictIntZ` — token-index variants for embedding models

**Never** ship a feature that only works with one effect system. If you add `predictX`, you must add `predictXZ`. If you add `fitX`, you must add `fitXZ`.

### 2. Close the Gap Equally
Default to implementing features for **both** DJL and DL4J backends simultaneously. Asymmetry is a last resort.

When a feature truly cannot be implemented in one backend:
- Document the technical reason (not just "not supported")
- Provide the equivalent escape-hatch path (ONNX export, raw framework API)
- Tag the limitation clearly in README coverage matrix

**Example**: Word2Vec training is DL4J-only because DJL is a model-serving engine with no NLP training primitives (no `SequenceVectors`, no SkipGram/CBOW). The escape hatch: train with PyTorch `nn.Embedding` → export ONNX → load via DJL.

**Anti-example**: Saying "DJL Embedding has no builder, use escape hatch" — we found `IdEmbedding` with a public builder. Always exhaust the API before giving up.

### 3. Semantic Versioning (SemVer) — `MAJOR.MINOR.PATCH`

Pre-1.0 (`0.y.z`) follows standard semver conventions:
- **MAJOR (y)**: New module, new LayerDef variants, DSL-breaking changes, API signature changes
- **MINOR (z)**: Bug fixes, doc-only changes, non-breaking internal refactors
- **PATCH**: Not used pre-1.0 (everything is y.z)

**Version bump checklist**:
- Adding a new `LayerDef` or `AdvancedLayerDef` variant → bump MAJOR (y)
- Adding a new top-level module (e.g., `embeddings/`) → bump MAJOR (y)
- Changing any existing method signature → bump MAJOR (y)
- Docs-only, test fixes, internal refactors → bump MINOR (z)

Post-1.0, standard semver applies: MAJOR for breaking changes, MINOR for new features, PATCH for fixes.

## Boundary Rules

- **Never** add framework-specific code to `core/`. It must stay backend-agnostic.
- **Never** use `asInstanceOf` or `Any` in `core/` unless type erasure forces it.
- **Never** leak DJL or DL4J types into the unified API (`ZModel.create`, `.predict`, `.fit`).
- **Never** suppress compiler warnings with `@nowarn` unless unavoidable Scala 3 quirk.
- **Always** preserve `Array[Array[Float]]` ↔ `Array[Float]` contract in unified API.
- **Always** mirror package structure: `zio.nn.djl` for DJL, `zio.nn.dl4j` for DL4J.
- **Always** add both `predictZ`/`fitZ` (ZIO) and `predict`/`fit` (Try) when adding new methods.
- **Always** export backend symbols into `zio.nn` via package object.

## Test Conventions

- Framework: **ZIO Test 2** (`ZIOSpecDefault`)
- Assertion style: `assertTrue(cond1, cond2, ...)` — no custom matchers
- Spec naming: `*Spec.scala` (not `*Test.scala`)
- Organization: mirror source package, nested `suite()` calls
- Test data: inline generation (no fixtures/factories)
- Resource cleanup: manual `close()` in Try-tests, `ZIO.scoped` / `.provideLayer(Scope.default)` in ZIO-tests

## User Communication Rules

- Prefer yes/no answers for confirmation questions.
- List options numerically when multiple choices exist.
- Default to reading test files first when asked about behavior.
- For bugs: show the failing assertion and the actual/expected values.
- For feature requests: ask "which backend?" before implementation.

## Explicit Context Checklist (new agent intake)

Before editing any file, verify:
- [ ] Which module(s) this affects (core, djl, dl4j, macros)
- [ ] Whether the change crosses backend boundaries (must touch both djl + dl4j)
- [ ] Whether the change touches the unified API (must keep Try + ZIO versions in sync)
- [ ] Whether tests exist for similar functionality (check matching *Spec.scala)
- [ ] Which Scala 3 feature is being used (context functions, enums, opaque types, etc.)
- [ ] Whether the change needs README.md or USER_GUIDE.md updates

## System Upgrade Notes

- CLI: `opencode` (no version pin)
- Scala: 3.x (cross-build compatible)
- SBT: latest (managed by version file)
- ZIO: 2.x (check build.sbt for exact version)
- DJL: ~0.36.x (check build.sbt)
- DL4J: 1.0.0-M2.1 (final milestone — no new releases expected from Konduit)
