package zio.nn

import zio.*

/** A ring-buffer pool of [[ReusableNativeBuffer]] slots.
  *
  * Acquire takes a buffer from the pool (semantically blocking if empty);
  * release clears and returns the buffer.
  *
  * {{{
  *   for {
  *     pool <- NativeBufferPool.create(slotCount = 4, slotCapacityBytes = 8192)
  *     buf  <- pool.acquire
  *     _    <- buf.putFloats(Array(1.0f, 2.0f))
  *     _    <- pool.release(buf)
  *     _    <- pool.close
  *   } yield ()
  * }}}
  *
  * Thread-safe — backed by ZIO [[zio.Queue.bounded]].
  */
final class NativeBufferPool private (
  queue: Queue[ReusableNativeBuffer]
):

  /** Take a buffer from the pool. Semantically blocks if all slots are
    * currently held by other consumers.
    */
  val acquire: UIO[ReusableNativeBuffer] =
    queue.take

  /** Clear and return a buffer to the pool. */
  def release(buf: ReusableNativeBuffer): UIO[Unit] =
    buf.clear() *> queue.offer(buf).unit

  /** Drain all buffers from the pool and close them, freeing off-heap memory.
    * Any future `acquire` will semantically block forever — create a new pool
    * instead of re-using a closed one.
    */
  val close: UIO[Unit] =
    queue.takeAll.flatMap(bufs => ZIO.foreachDiscard(bufs)(_.close()))

object NativeBufferPool:

  /** Create a pool with `slotCount` pre-allocated buffers.
    *
    * @param slotCount         number of ring-buffer slots
    * @param slotCapacityBytes per-slot ByteBuffer capacity
    */
  def create(
    slotCount: Int,
    slotCapacityBytes: Int
  ): UIO[NativeBufferPool] =
    for {
      bufs  <- ZIO.foreach(1 to slotCount)(_ => ReusableNativeBuffer.allocate(slotCapacityBytes))
      queue <- Queue.bounded[ReusableNativeBuffer](slotCount)
      _     <- queue.offerAll(bufs)
    } yield new NativeBufferPool(queue)

  /** Convenience — create a pool sized in floats per slot.
    *
    * @param slotCount     number of ring-buffer slots
    * @param floatsPerSlot per-slot float capacity
    */
  def createSlots(
    slotCount: Int,
    floatsPerSlot: Int
  ): UIO[NativeBufferPool] =
    create(slotCount, floatsPerSlot * 4)
