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
libraryDependencies += "io.github.szekai" %% "zio-nn-dl4j" % "0.8.0"   // DL4J
libraryDependencies += "io.github.szekai" %% "zio-nn-djl"  % "0.8.0"   // DJL
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
| `abs` | `a → Task[...]` |
| `square` | `a → Task[...]` |
| `sign` | `a → Task[...]` |
| `log` | `a → Task[...]` |
| `sigmoid` | `a → Task[...]` |
| `maximum` | `(a, b) → Task[...]` |
| `diagonal` | `a → Task[...]` |
| `get` | `(a, indices) → Task[...]` |
| `slice` | `(a, start, end) → Task[...]` |
| `concatenate` | `(a, b) → Task[...]` |
| `gather` | `(a, indices) → Task[...]` |
| `unique` | `a → Task[...]` |
| `countZeros` | `a → Task[Long]` |
| `std` | `a → Task[Float]` |
| `norm` | `a → Task[Float]` | L2 norm |
| `solve` | `(a, b, threshold: Double = 1e-12) → Task[Array[Float]]` | Solve `Ax = b`; singular values below `threshold` are zeroed |
| `lessThanOrEqual` | `(a, b) → Task[Array[Boolean]]` |
| `not` | `(a, threshold: Double = 1e-12) → Task[Array[Float]]` | Logical NOT; values ≤ `threshold` treated as true (zero) |
| `where` | `(cond, a, b, threshold: Double = 1e-12) → Task[Array[Float]]` | Element-wise `if (cond[i]) a[i] else b[i]` with threshold |

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
