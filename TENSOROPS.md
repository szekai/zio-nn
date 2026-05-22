# Tensor Operations Guide

## Quick Start

```scala
// DJL backend
import zio.nn.TensorOps.*
given NDManager = NDManager.newBaseManager()

val program = for
  a      <- create(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
  b      <- create(Array(Array(0.5f, 0.5f), Array(0.5f, 0.5f)))
  sum    <- add(a, b)              // element-wise addition
  diff   <- sub(a, b)              // subtraction
  prod   <- matMul(a, b)           // matrix multiplication
  arr    <- toDoubleArray(prod)    // NDArray → Array[Double]
yield arr
```

```scala
// DL4J backend (same API, different import)
import zio.nn.TensorOps.*
// No given NDManager needed for DL4J

val program = for
  a   <- createDouble(Array(Array(1.0, 2.0), Array(3.0, 4.0)))
  b   <- createDouble1D(Array(0.5, 0.5))
  s   <- sum(a)                       // reduce sum
  m   <- mean(a)                      // reduce mean
  d   <- toDoubleArray(m)
yield d
```

## All Operations

| Operation | DJL | DL4J | Signature |
|-----------|-----|------|-----------|
| `create` | ✅ | ✅ | `Array[Array[Float]] → Task[NDArray \| INDArray]` |
| `create1D` | ✅ | ✅ | `Array[Float] → Task[...]` |
| `createDouble` | ✅ | ✅ | `Array[Array[Double]] → Task[...]` |
| `createDouble1D` | ✅ | ✅ | `Array[Double] → Task[...]` |
| `add` | ✅ | ✅ | `(a, b) → Task[...]` |
| `sub` | ✅ | ✅ | `(a, b) → Task[...]` |
| `mul` | ✅ | ✅ | `(a, b) → Task[...]` |
| `div` | ✅ | ✅ | `(a, b) → Task[...]` |
| `matMul` | ✅ | ✅ | `(a, b) → Task[...]` |
| `dot` | ✅ | ✅ | `(a, b) → Task[...]` |
| `transpose` | ✅ | ✅ | `a → Task[...]` |
| `sum` | ✅ | ✅ | `a → Task[...]` |
| `mean` | ✅ | ✅ | `a → Task[...]` |
| `neg` | ✅ | ✅ | `a → Task[...]` |
| `toFloatArray` | ✅ | ✅ | `... → Task[Array[Float]]` |
| `toDoubleArray` | ✅ | ✅ | `... → Task[Array[Double]]` |
| `shape` | ✅ | ✅ | `... → Task[Array[Long]]` |

## DJL: NDManager required

DJL tensor ops need an implicit `NDManager`:

```scala
import zio.nn.TensorOps.*
given NDManager = NDManager.newBaseManager()
// Now create, add, matMul etc. work
```

## Escape Hatch

Need a raw framework operation not in TensorOps?

```scala
// DJL: use underlying NDArray directly
val nd: NDArray = ...
val result = nd.softmax(0)  // any NDArray method

// DL4J: use INDArray directly
val ind: INDArray = ...
val result = ind.reshape(2, 3)  // any INDArray method
```

Use `import zio.nn.implicits.*` to convert between unified and native types.
