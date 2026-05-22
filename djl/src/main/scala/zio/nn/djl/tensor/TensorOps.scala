package zio.nn.djl.tensor

import ai.djl.ndarray.{NDArray, NDManager}
import zio.*

/** ZIO-wrapped tensor math operations for DJL's NDArray.
  * Direct replacement for zenithfl's Nd4jZIO facade.
  *
  * NDManager is encapsulated internally — signatures match DL4J backend
  * exactly, enabling zero-code-change backend swap (just swap the JAR).
  *
  * Usage:
  * {{{
  *   import zio.nn.TensorOps.*
  *   for
  *     a <- createDouble1D(Array(1.0, 2.0, 3.0))
  *     b <- createDouble1D(Array(0.5, 0.5, 0.5))
  *     c <- add(a, b)
  *     d <- matMul(a, b)
  *     e <- toDoubleArray(c)
  *   yield e
  * }}}
  *
  * Advanced: use `scope.withNDManager` for explicit NDManager lifecycle
  * control over large batch operations.
  */
object TensorOps:

  // Process-scoped base manager. NDArrays created directly from the
  // base manager persist until shutdown(). Compute ops use temporary
  // sub-managers for intermediate allocations.
  private lazy val baseManager: NDManager = NDManager.newBaseManager()

  /** Release the base NDManager. Call before JVM exit in long-running apps. */
  def shutdown(): Unit = baseManager.close()

  def create(data: Array[Array[Float]]): Task[NDArray] =
    ZIO.attemptBlocking(baseManager.create(data))

  def create1D(data: Array[Float]): Task[NDArray] =
    ZIO.attemptBlocking(baseManager.create(data))

  def createDouble(data: Array[Array[Double]]): Task[NDArray] =
    ZIO.attemptBlocking(baseManager.create(data.map(_.map(_.toFloat))))

  def createDouble1D(data: Array[Double]): Task[NDArray] =
    ZIO.attemptBlocking(baseManager.create(data.map(_.toFloat)))

  // ═══ Compute ops (unchanged — already backend-agnostic) ═══

  def add(a: NDArray, b: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.add(b))

  def sub(a: NDArray, b: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.sub(b))

  def mul(a: NDArray, b: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.mul(b))

  def div(a: NDArray, b: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.div(b))

  def matMul(a: NDArray, b: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.matMul(b))

  def dot(a: NDArray, b: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.dot(b))

  def transpose(a: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.transpose())

  def sum(a: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.sum())

  def mean(a: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.mean())

  def neg(a: NDArray): Task[NDArray] =
    ZIO.attemptBlocking(a.neg())

  def toFloatArray(a: NDArray): Task[Array[Float]] =
    ZIO.attemptBlocking {
      val flat = if a.getShape.dimension() > 1 then a.flatten() else a
      val arr = new Array[Float](flat.size().toInt)
      for i <- arr.indices do arr(i) = flat.getFloat(i.toLong)
      arr
    }

  def toDoubleArray(a: NDArray): Task[Array[Double]] =
    ZIO.attemptBlocking {
      val flat = if a.getShape.dimension() > 1 then a.flatten() else a
      val arr = new Array[Double](flat.size().toInt)
      for i <- arr.indices do arr(i) = flat.getFloat(i.toLong).toDouble
      arr
    }

  def shape(a: NDArray): Task[Array[Long]] =
    ZIO.attemptBlocking(a.getShape.getShape)
