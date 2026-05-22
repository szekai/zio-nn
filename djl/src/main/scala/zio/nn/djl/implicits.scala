package zio.nn.djl

import ai.djl.ndarray.{NDList, NDArray, NDManager}

/** Implicit conversions between zio-nn unified types and DJL native types.
  * Import to use escape hatch without boilerplate:
  * {{{
  *   import zio.nn.djl.implicits.*
  *   val ndlist: NDList = myFloatArrays.toNDList
  *   val floats: Array[Array[Float]] = ndlist.toFloatArrays
  * }}}
  */
object implicits:

  extension (arrays: Array[Array[Float]])
    def toNDList(using NDManager): NDList =
      val mgr = summon[NDManager]
      val stacked = mgr.create(arrays)
      new NDList(stacked)

  extension (floats: Array[Float])
    def toNDArray(using NDManager): NDArray =
      summon[NDManager].create(floats)

  extension (ndlist: NDList)
    def toFloatArrays: Array[Array[Float]] =
      if ndlist.isEmpty then Array.empty
      else
        val head = ndlist.head()
        val total = head.size().toInt
        val arr = new Array[Float](total)
        for i <- 0 until total do arr(i) = head.getFloat(i.toLong)
        val shape = head.getShape
        val rows = if shape.dimension() > 0 then shape.get(0).toInt else 1
        val cols = if rows > 0 then total / rows else total
        arr.grouped(cols).toArray

  extension (ndarray: NDArray)
    def toFloatArray: Array[Float] =
      val arr = new Array[Float](ndarray.size().toInt)
      for i <- arr.indices do arr(i) = ndarray.getFloat(i.toLong)
      arr
