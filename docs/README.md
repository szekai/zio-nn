[← szekai.github.io](https://szekai.github.io/)

# zio-nn — Neural Network Library for ZIO

**Write once, run on any JVM deep learning framework. Swap the JAR, not the code.**

<span class="badge">Scala 3.8.3</span>
<span class="badge">ZIO 2.1</span>
<span class="badge">v0.6.0</span>

## Install

```scala
libraryDependencies += "io.github.szekai" %% "zio-nn-djl" % "0.6.0"
```

## Quick Start

```scala
import zio.nn.dsl.*
import zio.nn.*

val arch = Sequential(7)(LSTM(64, Tanh), Dense(32, ReLU), Output(1, MSE)).build
val model = ZModel.create(arch, "my-model").get
val pred = model.predict(features).get
model.fit(features, labels, epochs = 50)
model.close()
```

## ZIO-Native

```scala
import zio.nn.zioApi.*
ZIO.scoped {
  for
    model <- create(arch, "model")
    pred  <- model.predictZ(features)
  yield pred
}
```

## ZIO Stream

```scala
import zio.stream.*
// Live predictions from stream
priceStream.via(model.predictFlow).runCollect
// Online training from stream
dataStream.via(model.fitFlow(epochs = 1)).runDrain
```

## Backend Swap

```scala
"io.github.szekai" %% "zio-nn-djl"  % "0.6.0"  // ← swap
"io.github.szekai" %% "zio-nn-dl4j" % "0.6.0"  // ← zero code changes
```

## Features

| Feature | DJL | DL4J | Since |
|---------|-----|------|-------|
| `ZModel.create/predict/fit` | ✅ | ✅ | v0.1.0 |
| Architecture DSL | ✅ | ✅ | v0.1.0 |
| Conv2D / MaxPool2D / Flatten | ✅ | ✅ | v0.4.0 |
| FunctionalDef (DAG) | ✅ | ✅ | v0.5.3 |
| TensorOps (17 ops) | ✅ | ✅ | v0.4.0 |
| ZIO-native API | ✅ | ✅ | v0.3.0 |
| ZIO Stream (predictFlow, fitFlow) | ✅ | ✅ | v0.6.0 |
| Implicit conversions | ✅ | ✅ | v0.2.1 |
| `withNDManager` scope | ✅ | N/A | v0.4.1 |
| ONNX / TF / XGBoost engines | ✅ via DJL | N/A | v0.1.0 |

## Modules

| Module | Description | Tests |
|--------|-------------|-------|
| `zio-nn-core` | DSL + architecture types + implicits | 22 ✓ |
| `zio-nn-djl` | PyTorch / ONNX / TF / XGBoost + Stream | 6 ✓ |
| `zio-nn-dl4j` | JVM-native (no Python) + Stream | 22 ✓ |

## Links

- [GitHub](https://github.com/szekai/zio-nn) — source code
- [Full README](https://github.com/szekai/zio-nn/blob/main/README.md) — complete docs
- [TensorOps Guide](https://github.com/szekai/zio-nn/blob/main/TENSOROPS.md) — tensor operations
- [Publishing Guide](https://github.com/szekai/zio-nn/blob/main/PUBLISHING.md) — Maven Central
