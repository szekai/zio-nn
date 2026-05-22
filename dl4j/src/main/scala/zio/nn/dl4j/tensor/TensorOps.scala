package zio.nn.dl4j.tensor

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import zio.*

/** ZIO-wrapped tensor math operations for DL4J's INDArray.
  * Direct replacement for zenithfl's Nd4jZIO facade.
  *
  * Usage:
  * {{{
  *   import zio.nn.dl4j.tensor.TensorOps.*
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

  def create(data: Array[Array[Float]]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def create1D(data: Array[Float]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def createDouble(data: Array[Array[Double]]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def createDouble1D(data: Array[Double]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def add(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.add(b))

  def sub(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.sub(b))

  def mul(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.mul(b))

  def div(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.div(b))

  def matMul(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.mmul(b))

  def dot(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.mmul(b)) // dot product for 1D, mmul for 2D

  def transpose(a: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.transpose())

  def sum(a: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.sum(0))

  def mean(a: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.mean(0))

  def neg(a: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.neg())

  def toFloatArray(a: INDArray): Task[Array[Float]] =
    ZIO.attemptBlocking {
      val arr = new Array[Float](a.length().toInt)
      for i <- arr.indices do arr(i) = a.getFloat(i.toLong)
      arr
    }

  def toDoubleArray(a: INDArray): Task[Array[Double]] =
    ZIO.attemptBlocking {
      val arr = new Array[Double](a.length().toInt)
      for i <- arr.indices do arr(i) = a.getDouble(i.toLong)
      arr
    }

  def shape(a: INDArray): Task[Array[Long]] =
    ZIO.attemptBlocking(a.shape().map(_.toLong))
