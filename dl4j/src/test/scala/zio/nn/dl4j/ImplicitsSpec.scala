package zio.nn.dl4j

import zio.test.*
import zio.test.Assertion.*

object ImplicitsSpec extends ZIOSpecDefault:

  def spec = suite("DL4J Implicits")(
    test("Array[Array[Float]] → INDArray → Array[Array[Float]] roundtrip") {
      import implicits.*
      val original = Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
      val ind = original.toINDArray
      val back = ind.toFloatArrays
      assertTrue(back.length == 2, back(0).toSeq == Seq(1.0f, 2.0f))
    },
    test("Array[Float] → INDArray → Array[Float] roundtrip") {
      import implicits.*
      val original = Array(1.0f, 2.0f, 3.0f)
      val ind = original.toINDArray
      val back = ind.toFloatArray
      assertTrue(back.toSeq == original.toSeq)
    }
  )
