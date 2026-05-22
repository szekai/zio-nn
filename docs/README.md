[← szekai.github.io](https://szekai.github.io/)

# zio-nn — Neural Network Library for ZIO

**Write once, run on any JVM deep learning framework. Swap the JAR, not the code.**

```scala
// Define architecture — same for both backends
import zio.nn.dsl.*
val arch = Sequential(7)(LSTM(64, Tanh), Dense(32, ReLU), Output(1, MSE)).build

// Create model — same for both backends
import zio.nn.*
val model = ZModel.create(arch, "my-model").get

// Predict / Train — same for both backends
val predictions = model.predict(features).get      // Array[Array[Float]] → Array[Float]
model.fit(features, labels, epochs = 50)            // returns FitResult(loss, epochs)
model.close()
```

**Swap DJL ↔ DL4J by changing one line in `build.sbt` — zero code changes.**

## Install

```scala
// build.sbt
libraryDependencies += "io.github.szekai" %% "zio-nn-djl" % "0.5.4"
// or: libraryDependencies += "io.github.szekai" %% "zio-nn-dl4j" % "0.5.4"
```

## Modules

| Module | Description | Tests |
|--------|-------------|-------|
| `zio-nn-core` | DSL + architecture types + implicits | 22 ✓ |
| `zio-nn-djl` | PyTorch / ONNX / TF / XGBoost engines | 6 ✓ |
| `zio-nn-dl4j` | JVM-native — no Python deps | 22 ✓ |

## Quick Start

```scala
import zio.nn.dsl.*
import zio.nn.*

val arch = Sequential(7)(LSTM(64, Tanh), Dense(32, ReLU), Output(1, MSE)).build
val model = ZModel.create(arch, "my-model").get
model.fit(features, labels, epochs = 50)
val pred = model.predict(features).get
model.close()
```

## Backend Swap

```scala
"io.github.szekai" %% "zio-nn-djl"  % "0.5.4"  // ← swap this
"io.github.szekai" %% "zio-nn-dl4j" % "0.5.4"  // ← to this
```

## Features

| Feature | DJL | DL4J |
|---------|-----|------|
| `ZModel.create/predict/fit` | ✅ | ✅ |
| LSTM / Dense / Output | ✅ | ✅ |
| Conv2D / MaxPool2D / Flatten | ✅ | ✅ |
| TensorOps (17 ops) | ✅ | ✅ |
| ZIO-native API (zioApi) | ✅ | ✅ |
| Implicit conversions | ✅ | ✅ |
| Save / Load | ✅ | ✅ |
| FunctionalDef (DAG) | ✅ | ✅ |
| Multi-GPU / Spark | escape hatch | escape hatch |

## Links

- [GitHub](https://github.com/szekai/zio-nn)
- [Full README](https://github.com/szekai/zio-nn/blob/main/README.md)
- [TensorOps Guide](https://github.com/szekai/zio-nn/blob/main/TENSOROPS.md)
- [Publishing Guide](https://github.com/szekai/zio-nn/blob/main/PUBLISHING.md)
