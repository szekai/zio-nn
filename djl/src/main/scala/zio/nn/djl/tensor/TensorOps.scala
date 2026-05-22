package zio.nn.djl.tensor

import ai.djl.ndarray.{NDArray, NDManager}
import zio.*

/** ZIO-wrapped tensor math operations for DJL's NDArray.
  * Direct replacement for zenithfl's Nd4jZIO facade.
  *
  * Usage:
  * {{{
  *   import zio.nn.djl.tensor.TensorOps.*
  *   given NDManager = NDManager.newBaseManager()
  *   for
  *     a <- create(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
  *     b <- create(Array(Array(0.5f, 0.5f), Array(0.5f, 0.5f)))
  *     c <- add(a, b)
  *     d <- matMul(a, b)
  *     e <- toDoubleArray(c)
  *   yield e
  * }}}
  */
object TensorOps:

  def create(data: Array[Array[Float]])(using NDManager): Task[NDArray] =
    ZIO.attemptBlocking(summon[NDManager].create(data))

  def create1D(data: Array[Float])(using NDManager): Task[NDArray] =
    ZIO.attemptBlocking(summon[NDManager].create(data))

  def createDouble(data: Array[Array[Double]])(using NDManager): Task[NDArray] =
    ZIO.attemptBlocking(summon[NDManager].create(data.map(_.map(_.toFloat))))

  def createDouble1D(data: Array[Double])(using NDManager): Task[NDArray] =
    ZIO.attemptBlocking(summon[NDManager].create(data.map(_.toFloat)))

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
      val arr = new Array[Float](a.size().toInt)
      for i <- arr.indices do arr(i) = a.getFloat(i.toLong)
      arr
    }

  def toDoubleArray(a: NDArray): Task[Array[Double]] =
    ZIO.attemptBlocking {
      val arr = new Array[Double](a.size().toInt)
      for i <- arr.indices do arr(i) = a.getFloat(i.toLong).toDouble
      arr
    }

  def shape(a: NDArray): Task[Array[Long]] =
    ZIO.attemptBlocking(a.getShape.getShape)
