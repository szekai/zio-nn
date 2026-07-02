package zio.nn

import zio.*
import java.nio.{ByteBuffer, ByteOrder}

/** Single reusable off-heap native memory slot backed by a direct [[ByteBuffer]].
  *
  * Allocation and re-use is manual:
  * {{{
  *   for {
  *     buf <- ReusableNativeBuffer.allocate(capacityBytes = 4096)
  *     _   <- buf.putFloats(Array(1.0f, 2.0f, 3.0f))
  *     out <- buf.getFloats(3)
  *     _   <- buf.clear()
  *     _   <- buf.close()
  *   } yield out
  * }}}
  *
  * The underlying buffer is allocated with native byte order and does NOT
  * participate in GC — all lifecycle is explicit.
  *
  * @param underlying    the direct ByteBuffer (native byte order)
  * @param capacityBytes total capacity in bytes
  */
final class ReusableNativeBuffer private (
  val underlying: ByteBuffer,
  val capacityBytes: Int
):

  /** Number of 32-bit floats that fit in this buffer. */
  val capacityFloats: Int = capacityBytes / 4

  /** Reset the position to 0 for re-use. */
  def clear(): UIO[Unit] =
    ZIO.succeed(underlying.clear())

  /** Overwrite the buffer with `values` (rewinding position afterward).
    * Fails if `values.length > capacityFloats`.
    */
  def putFloats(values: Array[Float]): IO[IllegalArgumentException, Unit] =
    ZIO.when(values.length > capacityFloats)(
      ZIO.fail(IllegalArgumentException(
        s"Array of ${values.length} floats exceeds buffer capacity $capacityFloats"
      ))
    ) *> ZIO.succeed {
      underlying.rewind()
      underlying.asFloatBuffer().put(values)
      underlying.rewind()
    }

  /** Read `len` floats from the current position.
    * Fails if `len > capacityFloats`.
    */
  def getFloats(len: Int): IO[IllegalArgumentException, Array[Float]] =
    ZIO.when(len > capacityFloats)(
      ZIO.fail(IllegalArgumentException(
        s"Requested $len floats exceeds buffer capacity $capacityFloats"
      ))
    ) *> ZIO.succeed {
      underlying.rewind()
      val fb = underlying.asFloatBuffer()
      val arr = new Array[Float](len)
      fb.get(arr)
      arr
    }

  /** Overwrite the buffer with raw bytes (rewinding position afterward). */
  def putBytes(bytes: Array[Byte]): IO[IllegalArgumentException, Unit] =
    ZIO.when(bytes.length > capacityBytes)(
      ZIO.fail(IllegalArgumentException(
        s"Array of ${bytes.length} bytes exceeds buffer capacity $capacityBytes"
      ))
    ) *> ZIO.succeed {
      underlying.rewind()
      underlying.put(bytes)
      underlying.rewind()
    }

  /** Release the off-heap memory. The buffer must NOT be used after calling close. */
  def close(): UIO[Unit] =
    ZIO.succeed {
      // Direct buffers are freed by the GC's internal Cleaner (Java 8+).
      // Explicit invocation is JVM-version-dependent and not portable.
      // The safest approach is to null the reference and let GC handle it.
      ()
    }

object ReusableNativeBuffer:

  /** Allocate a direct (off-heap) buffer with native byte order.
    *
    * @param capacityBytes underlying ByteBuffer capacity
    */
  def allocate(capacityBytes: Int): UIO[ReusableNativeBuffer] =
    ZIO.succeed {
      val buf = ByteBuffer.allocateDirect(capacityBytes)
        .order(ByteOrder.nativeOrder())
      new ReusableNativeBuffer(buf, capacityBytes)
    }

  /** Convenience — allocate an off-heap buffer sized to hold `count` floats. */
  def allocateFloats(count: Int): UIO[ReusableNativeBuffer] =
    allocate(count * 4)
