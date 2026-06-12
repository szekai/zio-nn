## zio-nn

Write-once neural network library for ZIO. Swap DJL ↔ DL4J by changing one JAR — zero code changes.

## Scoring Matrix

| Directory | Scala Sources | Tests | Score | AGENTS.md |
|-----------|--------------|-------|-------|-----------|
| root      | (config/build) | — | HIGH | ✅ (this file) |
| core/     | 4 (ModelArchitecture, dsl, ConfigLoader, Preprocessing) | 2 files, 22 tests | HIGH | ✅ |
| djl/      | 8 (Backend, implicits, scope, tensor/, wrappers, zio, ZTokenizer, ImageTransformer) | 6 files, 20 tests | HIGH | ✅ |
| dl4j/     | 7 (Backend, implicits, tensor/, wrappers, zio, ZTokenizer, ImageTransformer) | 8 files, 37 tests | HIGH | ✅ |
| embeddings/ | 1 (Word2Vec) | 0 files, 0 tests | MEDIUM | ❌ |
| macros/   | 0 | 0 | LOW | ❌ |
| project/  | 0 (SBT config only) | 0 | LOW | ❌ |

## Architecture Overview

- **core/** — Framework-agnostic: `ModelDef`, `LayerDef`, `ActivationFn`, `LossFn`, `OptimizerDef`, DSL (`dsl.*`), `ConfigLoader`, `TokenizerConfig`, `EncodingResult`, `ImageTransformDef`, `ImagePipeline`
- **djl/** — Deep Java Library backend: `ZModel` wrapping `ZooModel`, `Backend.compile` → DJL `Block`, `TensorOps` for NDArray math, `scope.withNDManager` for resource mgmt, `ZTokenizer` (HuggingFace), `ImageTransformer` (DJL CV)
- **dl4j/** — DeepLearning4j backend: `ZModel` wrapping `MultiLayerNetwork`, `Backend.compile` → DL4J `MultiLayerNetwork`, `TensorOps` for INDArray math, `ZTokenizer` (regex/whitespace), `ImageTransformer` (ND4J)

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

### 3. Versioning — `sbt-dynver` (git-tag-driven)

Version is managed automatically by `sbt-dynver` (bundled via `sbt-ci-release`):

- **Current version**: derived from the nearest `v*` git tag (e.g., `v0.8.0` → version `0.8.0`)
- **Untagged commits**: produce `<tag>-<n>-g<sha>-SNAPSHOT` (e.g., `0.8.0-3-gdeadbeef-SNAPSHOT`)
- **No tags at all**: falls back to `0.1.0-SNAPSHOT` (you should tag before first release)
- Pre-1.0 semver: MAJOR (y) = new modules/layers/breaking changes, MINOR (z) = bug fixes/docs

**Version bump** = push a tag matching `v<major>.<minor>.<patch>`:

```bash
git tag v0.9.0            # bump MAJOR (new module or breaking change)
git tag v0.8.1            # bump MINOR (bug fixes, docs, refactors)
git push origin v0.9.0    # triggers CI release
```

**Tag checklist** (bump MAJOR y):
- Adding a new `LayerDef` or `AdvancedLayerDef` variant
- Adding a new top-level module (e.g., `embeddings/`)
- Changing any existing method signature

**Tag checklist** (bump MINOR z):
- Docs-only, test fixes, non-breaking internal refactors

### 3a. Release Candidates (`-RC`)

Tag with `-RC<N>` suffix for pre-release testing (semver pre-release precedence means `0.9.0-RC1` sorts before `0.9.0`):

```bash
git tag v0.9.0-RC1
git push origin v0.9.0-RC1    # CI publishes to Sonatype, but users get RC artifacts
# iterate with -RC2, -RC3 if needed …
git tag v0.9.0                # final release
git push origin v0.9.0
```

**Why RC**: Maven Central artifacts are immutable. An RC lets you verify the published artifact in real projects before committing to the final version. If an RC has bugs, burn `-RC2` instead of `0.9.1`.

**When**: Required for any release that adds a new module or LayerDef variant. Optional for minor (z) bumps.

### 3b. Milestones

Use GitHub Milestones to group issues and track progress toward a version:

```
Milestone: v0.9.0
  ├── #15  Transformer / Attention layers
  ├── #16  Model checkpointing improvements
  └── #17  Performance benchmarks
```

**Convention**: Milestone title matches the target version tag. Close the milestone when the version is released. Each open issue in a milestone blocks that release.

### 3c. CI/CD Pipeline (`sbt-ci-release` + GitHub Actions)

The CI workflow (`.github/workflows/ci.yml`) has two jobs:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `test` | PR + push to `main` | `sbt test` — compile and run all 50 tests across all modules |
| `release` | Push tag `v*` (after `test` passes) | `sbt ci-release` — tags the release, publishes signed artifacts to Sonatype, closes and releases the staging repository |

**Required secrets** (set in GitHub repo settings → Secrets and variables → Actions):
- `SONATYPE_USERNAME` / `SONATYPE_PASSWORD` — Sonatype (s01.oss.sonatype.org) credentials
- `PGP_SECRET` — ASCII-armored GPG private key (`gpg --armor --export-secret-keys <key-id>`)
- `PGP_PASSPHRASE` — passphrase for the GPG key

**Release workflow**:
1. Ensure `main` is green (all tests pass)
2. `git checkout main && git pull --tags`
3. `git tag v<version>` (e.g., `v0.9.0`)
4. `git push origin v<version>`
5. GitHub Actions runs `test` → `release` → artifact lands on Maven Central (~10 min)
6. Create a GitHub Release from the tag for visibility

**Local testing** (without CI):
```bash
# publishTo and Sonatype staging are managed internally by sbt-ci-release
CI=true sbt ci-release                     # full ci-release flow locally
```

**Note**: `build.sbt` must NOT define `version`, `publishTo`, `publishMavenStyle`, or `credentials` — sbt-ci-release handles all of them.

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
