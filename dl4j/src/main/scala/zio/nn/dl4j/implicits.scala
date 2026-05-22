package zio.nn.dl4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/** Implicit conversions between zio-nn unified types and DL4J native types.
  * Import to use escape hatch without boilerplate:
  * {{{
  *   import zio.nn.dl4j.implicits.*
  *   val indarray: INDArray = myFloatArrays.toINDArray
  *   val floats: Array[Array[Float]] = indarray.toFloatArrays
  * }}}
  */
object implicits:

  extension (arrays: Array[Array[Float]])
    def toINDArray: INDArray = Nd4j.create(arrays)

  extension (floats: Array[Float])
    def toINDArray: INDArray = Nd4j.create(floats)

  extension (indarray: INDArray)
    def toFloatArrays: Array[Array[Float]] =
      val rows = indarray.rows().toInt
      val cols = indarray.columns().toInt
      Array.tabulate(rows) { i =>
        val row = new Array[Float](cols)
        for j <- 0 until cols do row(j) = indarray.getFloat(i.toLong, j.toLong)
        row
      }

    def toFloatArray: Array[Float] =
      val arr = new Array[Float](indarray.length().toInt)
      for i <- arr.indices do arr(i) = indarray.getFloat(i.toLong)
      arr
