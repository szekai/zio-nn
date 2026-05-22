package zio.nn.djl

import zio.test.*
import zio.test.Assertion.*

object ImplicitsSpec extends ZIOSpecDefault:

  def spec = suite("DJL Implicits")(
    test("toFloatArray roundtrip preserves values") {
      import implicits.*
      given ai.djl.ndarray.NDManager = ai.djl.ndarray.NDManager.newBaseManager()
      val original = Array(1.0f, 2.0f, 3.0f)
      val nd = original.toNDArray
      val back = nd.toFloatArray
      assertTrue(back.sameElements(original))
    }
  )
