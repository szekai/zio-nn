# zio-nn — Neural Network Library for ZIO

> **Write once, run on any JVM deep learning framework. Swap the JAR, not the code.**

## Quick Links

| Document | Description |
|----------|-------------|
| [📖 User Guide](USER_GUIDE.md) | Architecture, DSL reference, escape hatches, backend swap guide |
| [🔢 Tensor Ops](TENSOROPS.md) | Full tensor operation reference and examples |
| [📦 Publishing](PUBLISHING.md) | Maven Central release process |
| [📘 Scaladoc API](api/index.html) | Auto-generated API documentation |

## About

`zio-nn` provides a framework-agnostic neural network DSL for Scala 3 + ZIO.
Define your architecture once — run on DJL (PyTorch/ONNX/TF) or DL4J by swapping one line in `build.sbt`.

```scala
import zio.nn.dsl.*
val arch = Sequential(7)(LSTM(64, Tanh), Dense(32, ReLU), Output(1, MSE)).build

import zio.nn.dl4j.ZModel   // or zio.nn.djl.ZModel for DJL
val model = ZModel.create(arch, "my-model").get
```

## Repositories

- **GitHub**: [github.com/szekai/zio-nn](https://github.com/szekai/zio-nn)
- **Maven Central**: `io.github.szekai` %% `zio-nn-core` / `zio-nn-djl` / `zio-nn-dl4j`
