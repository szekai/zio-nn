package zio.nn.storch

import torch.*
import torch.DType.*

/** Implicit conversions between zio-nn unified types and storch native types.
  * Import to use escape hatch without boilerplate:
  * {{{
  *   import zio.nn.storch.implicits.*
  *   val tensor: Tensor[Float32] = myFloats.toTensor
  *   val floats: Array[Float] = tensor.toFloatArray
  * }}}
  */
object implicits:

  extension (floats: Array[Float])
    def toTensor: Tensor[Float32] =
      torch.Tensor(floats)

  extension (arrays: Array[Array[Float]])
    def toTensor: Tensor[Float32] =
      val flattened = arrays.flatten
      val batchSize = arrays.length
      val cols = if batchSize > 0 then arrays(0).length else 0
      torch.Tensor(flattened).reshape(batchSize, cols)

  extension (tensor: Tensor[Float32])
    def toFloatArray: Array[Float] =
      tensor.toArray

    def toFloatArrays: Array[Array[Float]] =
      val shape = tensor.shape
      val total = tensor.numel.toInt
      val rows = if shape.length > 0 then shape(0).toInt else 1
      val cols = if rows > 0 then total / rows else total
      tensor.toArray.grouped(cols).toArray

  extension (seq: Seq[Float])
    def toTensor: Tensor[Float32] =
      torch.Tensor(seq.toArray)

  extension (tensor: Tensor[Float32])
    def toFloatSeq: Seq[Float] =
      tensor.toFloatArray.toSeq
