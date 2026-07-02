package zio.nn

import zio.*
import zio.test.*
import zio.test.Assertion.*

object ReusableNativeBufferSpec extends ZIOSpecDefault:

  def spec = suite("ReusableNativeBuffer")(
    // ── Buffer allocate / capacity ──
    test("allocate returns buffer with correct byte capacity") {
      for
        buf <- ReusableNativeBuffer.allocate(1024)
        _   <- buf.close()
      yield assertTrue(buf.capacityBytes == 1024)
    },

    test("allocateFloats returns buffer with correct float capacity") {
      for
        buf <- ReusableNativeBuffer.allocateFloats(64)
        _   <- buf.close()
      yield assertTrue(buf.capacityFloats == 64, buf.capacityBytes == 256)
    },

    // ── putFloats / getFloats round-trip ──
    test("putFloats then getFloats round-trips correctly") {
      for
        buf <- ReusableNativeBuffer.allocateFloats(16)
        _   <- buf.putFloats(Array(1.0f, 2.0f, 3.0f, 4.0f))
        out <- buf.getFloats(4)
        _   <- buf.close()
      yield assertTrue(
        out(0) == 1.0f, out(1) == 2.0f, out(2) == 3.0f, out(3) == 4.0f,
        out.length == 4
      )
    },

    test("putFloats after clear returns correct values") {
      for
        buf <- ReusableNativeBuffer.allocateFloats(8)
        _   <- buf.putFloats(Array(10.0f, 20.0f))
        _   <- buf.clear()
        _   <- buf.putFloats(Array(30.0f, 40.0f))
        out <- buf.getFloats(2)
        _   <- buf.close()
      yield assertTrue(out(0) == 30.0f, out(1) == 40.0f)
    },

    test("clear does not change buffer capacity") {
      for
        buf <- ReusableNativeBuffer.allocate(1024)
        _   <- buf.clear()
        cap <- ZIO.succeed(buf.capacityBytes)
        _   <- buf.close()
      yield assertTrue(cap == 1024)
    },

    test("putFloats overflow fails") {
      for
        buf  <- ReusableNativeBuffer.allocateFloats(2)
        exit <- buf.putFloats(Array(1.0f, 2.0f, 3.0f)).exit
        _    <- buf.close()
      yield assertTrue(exit.isFailure)
    },

    test("getFloats overflow fails") {
      for
        buf  <- ReusableNativeBuffer.allocateFloats(2)
        exit <- buf.getFloats(99).exit
        _    <- buf.close()
      yield assertTrue(exit.isFailure)
    },

    // ── putBytes / getFloats interop ──
    test("putBytes then getFloats round-trips correctly") {
      for
        buf <- ReusableNativeBuffer.allocate(16)  // 16 bytes = 4 floats
        _   <- buf.putBytes(Array[Byte](0, 0, 0x40, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        out <- buf.getFloats(1)
        _   <- buf.close()
      yield assertTrue(math.abs(out(0) - 3.0f) < 1e-6f)
    },

    test("putBytes overflow fails") {
      for
        buf  <- ReusableNativeBuffer.allocate(4)
        exit <- buf.putBytes(Array[Byte](1, 2, 3, 4, 5)).exit
        _    <- buf.close()
      yield assertTrue(exit.isFailure)
    },

    // ── Pool ──
    test("pool acquire / release / acquire round-trips") {
      for
        pool <- NativeBufferPool.create(slotCount = 2, slotCapacityBytes = 64)
        a    <- pool.acquire
        _    <- a.putFloats(Array(1.0f, 2.0f))
        _    <- pool.release(a)
        b    <- pool.acquire
        out  <- b.getFloats(2)
        _    <- pool.close
      yield assertTrue(out(0) == 0.0f, out(1) == 0.0f) // cleared on release
    },

    test("pool acquire blocks when empty and unblocks on release") {
      for
        pool <- NativeBufferPool.create(slotCount = 1, slotCapacityBytes = 64)
        a    <- pool.acquire
        // a second acquire in a separate fiber should block until a is released
        f    <- pool.acquire.fork
        _    <- pool.release(a)  // unblocks f
        b    <- f.join
        _    <- pool.close
      yield assertTrue(b.capacityBytes == 64)
    },

    test("pool acquire from multiple fibers concurrently") {
      for
        pool  <- NativeBufferPool.create(slotCount = 2, slotCapacityBytes = 64)
        bufs  <- ZIO.foreachPar(List("write-1", "write-2")) { _ =>
          for
            buf <- pool.acquire
            _   <- buf.putFloats(Array(42.0f))
            _   <- pool.release(buf)
          yield buf
        }
        _     <- pool.close
      yield assertTrue(bufs.size == 2)
    },

    test("pool close drains all buffers") {
      for
        pool <- NativeBufferPool.create(slotCount = 3, slotCapacityBytes = 32)
        a    <- pool.acquire
        b    <- pool.acquire
        _    <- pool.release(a)
        _    <- pool.release(b)
        _    <- pool.close
        res  <- pool.acquire.timeout(Duration.fromMillis(50))
      yield assertTrue(res.isEmpty)
    } @@ TestAspect.withLiveClock
  )
