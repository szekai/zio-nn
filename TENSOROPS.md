# Tensor Operations Guide

## Quick Start

Identical API for both backends. Swap the JAR, not the code.

```scala
import zio.nn.TensorOps.*   // resolves to DL4J or DJL backend automatically

val program = for
  a   <- createDouble1D(Array(1.0, 2.0, 3.0, 4.0))
  b   <- createDouble1D(Array(0.5, 0.5, 0.5, 0.5))
  sum <- add(a, b)              // element-wise addition
  m   <- matMul(a, b)           // matrix multiplication
  arr <- toDoubleArray(m)       // native → Array[Double]
yield arr
```

**No `given NDManager` needed** — both backends have identical, implicit-free signatures.

## Backend Swap

```scala
// build.sbt — swap this ONE line:
libraryDependencies += "dev.zio" %% "zio-nn-dl4j" % "0.5.1"   // DL4J
libraryDependencies += "dev.zio" %% "zio-nn-djl"  % "0.5.1"   // DJL
```

All consumer code remains unchanged. `import zio.nn.TensorOps.*` auto-resolves.

## All Operations

| Operation | Signature |
|-----------|-----------|
| `create` | `Array[Array[Float]] → Task[...]` |
| `create1D` | `Array[Float] → Task[...]` |
| `createDouble` | `Array[Array[Double]] → Task[...]` |
| `createDouble1D` | `Array[Double] → Task[...]` |
| `add` | `(a, b) → Task[...]` |
| `sub` | `(a, b) → Task[...]` |
| `mul` | `(a, b) → Task[...]` |
| `div` | `(a, b) → Task[...]` |
| `matMul` | `(a, b) → Task[...]` |
| `dot` | `(a, b) → Task[...]` |
| `transpose` | `a → Task[...]` |
| `sum` | `a → Task[...]` |
| `mean` | `a → Task[...]` |
| `neg` | `a → Task[...]` |
| `toFloatArray` | `... → Task[Array[Float]]` |
| `toDoubleArray` | `... → Task[Array[Double]]` |
| `shape` | `... → Task[Array[Long]]` |

## DJL: Advanced NDManager Control

For batch operations where you want explicit NDManager lifecycle:

```scala
import zio.nn.djl.scope.*

scope.withNDManager {
  for
    a <- TensorOps.createDouble1D(data)  // uses scoped manager
    b <- TensorOps.createDouble1D(more)
    c <- TensorOps.add(a, b)
  yield c
}  // NDManager auto-closed by ZIO Scope
```

Without `scope.withNDManager`, DJL uses an internal base manager with
per-call sub-managers that auto-close — safe for all workloads.

## Escape Hatch

Need a raw framework operation not in TensorOps?

```scala
// DJL: use underlying NDArray directly
val nd: NDArray = ???
val result = nd.softmax(0)

// DL4J: use INDArray directly
val ind: INDArray = ???
val result = ind.reshape(2, 3)
```

Use `import zio.nn.implicits.*` to convert between unified and native types.
