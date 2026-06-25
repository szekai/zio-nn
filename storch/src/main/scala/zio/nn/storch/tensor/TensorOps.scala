package zio.nn.storch.tensor

import torch.*
import torch.DType.*
import zio.*
import scala.util.Try

/** Tensor math operations for storch Tensor[Float32], wrapping common
  * operations in Try for consistent error handling.
  *
  * Mirrors the DJL and DL4J TensorOps pattern — each op returns
  * `Try[Tensor[Float32]]`, with ZIO variants (`*Z`) returning
  * `Task[Tensor[Float32]]`.
  *
  * Usage:
  * {{{
  *   import zio.nn.storch.tensor.TensorOps.*
  *   for
  *     a     <- add(ta, tb)
  *     b     <- matmul(ta, tb)
  *     flat  <- toArray(a)
  *   yield flat
  * }}}
  *
  * ZIO variant:
  * {{{
  *   for
  *     a     <- addZ(ta, tb)
  *     b     <- matmulZ(ta, tb)
  *   yield b
  * }}}
  */
object TensorOps:

  // ═══ Element-wise arithmetic ═══

  /** Element-wise addition: a + b */
  def add(a: Tensor[Float32], b: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(a + b)

  /** Element-wise subtraction: a - b */
  def sub(a: Tensor[Float32], b: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(a - b)

  /** Element-wise multiplication: a * b */
  def mul(a: Tensor[Float32], b: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(a * b)

  /** Element-wise division: a / b */
  def div(a: Tensor[Float32], b: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(a / b)

  // ═══ Linear algebra ═══

  /** Matrix multiplication: a.matmul(b) */
  def matmul(a: Tensor[Float32], b: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(a.matmul(b))

  // ═══ Shape operations ═══

  /** Reshape tensor to new shape. Total elements must match. */
  def reshape(t: Tensor[Float32], shape: Array[Int]): Try[Tensor[Float32]] =
    Try(t.reshape(shape*))

  /** Transpose two dimensions.
    * storch transpose takes exactly two dims: t.transpose(dim0, dim1).
    */
  def transpose(t: Tensor[Float32], dims: Array[Int]): Try[Tensor[Float32]] =
    Try(t.transpose(dims(0), dims(1)))

  /** Remove dimension of size 1 at position dim. */
  def squeeze(t: Tensor[Float32], dim: Int): Try[Tensor[Float32]] =
    Try(t.squeeze(dim))

  /** Add dimension of size 1 at position dim. */
  def unsqueeze(t: Tensor[Float32], dim: Int): Try[Tensor[Float32]] =
    Try(t.unsqueeze(dim))

  /** Repeat tensor along given dimensions.
    * Falls back to broadcastTo if repeat is unavailable.
    */
  def tile(t: Tensor[Float32], repeats: Array[Int]): Try[Tensor[Float32]] =
    Try(t.repeat(repeats*))

  /** Gather values along a dimension given index tensor. */
  def gather(t: Tensor[Float32], dim: Int, index: Tensor[torch.Int64]): Try[Tensor[Float32]] =
    Try(t.gather(dim.toLong, index))

  // ═══ Combine operations ═══

  /** Stack a sequence of tensors along a new dimension. */
  def stack(tensors: Seq[Tensor[Float32]], dim: Int): Try[Tensor[Float32]] =
    Try(torch.stack(tensors, dim))

  /** Concatenate a sequence of tensors along an existing dimension. */
  def concat(tensors: Seq[Tensor[Float32]], dim: Int): Try[Tensor[Float32]] =
    Try(torch.cat(tensors, dim))

  // ═══ Reduction operations ═══

  /** Sum all elements of the tensor. */
  def sum(t: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(t.sum())

  /** Mean of all elements of the tensor. */
  def mean(t: Tensor[Float32]): Try[Tensor[Float32]] =
    Try(t.mean())

  // ═══ Creation operations ═══

  /** Create a tensor filled with zeros. */
  def zeros(shape: Array[Int]): Try[Tensor[Float32]] =
    Try(torch.zeros[Float32](shape*))

  /** Create a tensor filled with ones. */
  def ones(shape: Array[Int]): Try[Tensor[Float32]] =
    Try(torch.ones[Float32](shape*))

  // ═══ Data extraction ═══

  /** Extract float array from a tensor. */
  def toArray(t: Tensor[Float32]): Try[Array[Float]] =
    Try(t.toArray)

  // ═══ ZIO variants ═══

  def addZ(a: Tensor[Float32], b: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(add(a, b))

  def subZ(a: Tensor[Float32], b: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(sub(a, b))

  def mulZ(a: Tensor[Float32], b: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(mul(a, b))

  def divZ(a: Tensor[Float32], b: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(div(a, b))

  def matmulZ(a: Tensor[Float32], b: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(matmul(a, b))

  def reshapeZ(t: Tensor[Float32], shape: Array[Int]): Task[Tensor[Float32]] =
    ZIO.fromTry(reshape(t, shape))

  def transposeZ(t: Tensor[Float32], dims: Array[Int]): Task[Tensor[Float32]] =
    ZIO.fromTry(transpose(t, dims))

  def squeezeZ(t: Tensor[Float32], dim: Int): Task[Tensor[Float32]] =
    ZIO.fromTry(squeeze(t, dim))

  def unsqueezeZ(t: Tensor[Float32], dim: Int): Task[Tensor[Float32]] =
    ZIO.fromTry(unsqueeze(t, dim))

  def tileZ(t: Tensor[Float32], repeats: Array[Int]): Task[Tensor[Float32]] =
    ZIO.fromTry(tile(t, repeats))

  def gatherZ(t: Tensor[Float32], dim: Int, index: Tensor[torch.Int64]): Task[Tensor[Float32]] =
    ZIO.fromTry(gather(t, dim, index))

  def stackZ(tensors: Seq[Tensor[Float32]], dim: Int): Task[Tensor[Float32]] =
    ZIO.fromTry(stack(tensors, dim))

  def concatZ(tensors: Seq[Tensor[Float32]], dim: Int): Task[Tensor[Float32]] =
    ZIO.fromTry(concat(tensors, dim))

  def sumZ(t: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(sum(t))

  def meanZ(t: Tensor[Float32]): Task[Tensor[Float32]] =
    ZIO.fromTry(mean(t))

  def zerosZ(shape: Array[Int]): Task[Tensor[Float32]] =
    ZIO.fromTry(zeros(shape))

  def onesZ(shape: Array[Int]): Task[Tensor[Float32]] =
    ZIO.fromTry(ones(shape))

  def toArrayZ(t: Tensor[Float32]): Task[Array[Float]] =
    ZIO.fromTry(toArray(t))

end TensorOps
