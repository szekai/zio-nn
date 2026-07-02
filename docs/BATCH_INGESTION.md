# High-Throughput Batch Ingestion

## Problem

During training and inference, each batch of data must be transferred from Java
heap memory to the native (off-heap) memory region used by the backend tensor
library (DJL NDArray, DL4J INDArray, PyTorch Tensor). When this happens for
*every* batch — thousands of times over a training run — GC pressure mounts,
pauses increase, and throughput drops.

The JVM's garbage collector has no visibility into off-heap memory. Each
`ByteBuffer.allocateDirect(...)` call also incurs a JNI upcall. By pooling
off-heap buffers, we eliminate both allocation overhead and GC churn.

## Solution

Two components, both in the **core** module (framework-agnostic):

### 1. `ReusableNativeBuffer`

A single off-heap `ByteBuffer` slot with manual lifecycle:

| Method | Returns | Behaviour |
|--------|---------|-----------|
| `allocate(capacityBytes)` | `UIO[ReusableNativeBuffer]` | Allocate a direct buffer |
| `clear()` | `UIO[Unit]` | Reset position to 0 |
| `putFloats(values)` | `IO[IllegalArg, Unit]` | Overwrite with float array |
| `getFloats(len)` | `IO[IllegalArg, Array[Float]]` | Read back float array |
| `putBytes(bytes)` | `IO[IllegalArg, Unit]` | Overwrite with byte array |
| `close()` | `UIO[Unit]` | Release off-heap memory |

### 2. `NativeBufferPool`

A ring-buffer pool of N pre-allocated `ReusableNativeBuffer` slots backed by
`zio.Queue.bounded`:

```scala
val pool: UIO[NativeBufferPool] = NativeBufferPool.create(slotCount = 4, slotCapacityBytes = 8192)

// In a consumer fiber:
for {
  buf <- pool.acquire          // blocks until a slot is free
  _   <- buf.putFloats(samples)
  _   <- pool.release(buf)     // clears and returns to pool
} yield ()
```

## Usage Pattern

### With `DataSetLoader` (typical pipeline)

```scala
for {
  pool <- NativeBufferPool.create(slotCount = 4, slotCapacityBytes = batchSize * 4)
  loader <- DataSetLoader.fromImageDir(imgDir, pipeline, batchSize) { (bytes, pl) =>
    Try {
      // Decode → pool.acquire → fill → return reference
    }
  }
  _ <- loader.fitFlow(model, ...).runForeach { batch =>
    for {
      buf <- pool.acquire
      _   <- buf.putFloats(batch._1.flatten) // copy to native
      _   <- pool.release(buf)
    } yield ()
  }
  _ <- pool.close
} yield ()
```

## Zero-Copy Backend Integration

The bridge helpers live in the **dl4j/** and **djl/** modules respectively.
These are optional — not part of the core module API.

### Key Rule

A `ReusableNativeBuffer` wraps a **direct** `ByteBuffer` with `nativeOrder()`.
The backend must accept the underlying pointer without copying:

| Backend | View over `ByteBuffer` | Pitfall |
|---------|------------------------|---------|
| DL4J (ND4J) | `NioDataBuffer` wrapping `buf.underlying` | ND4J uses C-order by default; DJL uses C-order natively |
| DJL | `NDArray` via `NDManager.create(buf.underlying)` | `NDArray` may internally copy if shape is incompatible |

### DL4J Bridge

```scala
// dl4j/src/main/scala/zio/nn/dl4j/ZeroCopyBridge.scala
def wrapAsINDArray(buf: ReusableNativeBuffer, shape: Array[Long]): INDArray = {
  val ptr = new NioDataBuffer(
    buf.underlying,
    DataType.FLOAT,
    buf.capacityFloats, 0L)
  Nd4j.create(ptr, shape, Order.C)
}
```

### DJL Bridge

```scala
// djl/src/main/scala/zio/nn/djl/ZeroCopyBridge.scala
def wrapAsNDArray(buf: ReusableNativeBuffer, manager: NDManager, shape: Shape): NDArray = {
  manager.create(buf.underlying, shape)
}
```

**Important**: DJL and ND4J copies are best-effort. Always profile to confirm
zero-copy is actually achieved for your specific shape and data layout.

## Performance Considerations

- **Slot count**: Start with `2×` the number of concurrent consumer fibers.
  For a single threaded `fitFlow`, 2–4 slots suffice.
- **Buffer size**: Align to `batchSize × featureDim × 4` (since one float = 4 bytes).
- **GC impact**: Monitor with `-XX:NativeMemoryTracking=summary` and
  `jcmd <pid> VM.native_memory summary`.
- **Thread safety**: `ReusableNativeBuffer` is **not** thread-safe —
  protect it via `NativeBufferPool.acquire`/`release` or external synchronization.

## See Also

- `core/src/main/scala/zio/nn/ReusableNativeBuffer.scala`
- `core/src/main/scala/zio/nn/NativeBufferPool.scala`
- `dl4j/src/main/scala/zio/nn/dl4j/ZeroCopyBridge.scala`
- `djl/src/main/scala/zio/nn/djl/ZeroCopyBridge.scala`
