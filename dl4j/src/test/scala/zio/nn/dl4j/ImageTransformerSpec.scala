package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import java.awt.image.BufferedImage

object ImageTransformerSpec extends ZIOSpecDefault:

  private def createTestImageBytes(width: Int, height: Int, gray: Boolean = false): Array[Byte] =
    val imgType = if gray then BufferedImage.TYPE_BYTE_GRAY else BufferedImage.TYPE_INT_RGB
    val img = BufferedImage(width, height, imgType)
    val g = img.createGraphics
    g.setColor(java.awt.Color.WHITE)
    g.fillRect(0, 0, width, height)
    g.dispose()
    val baos = ByteArrayOutputStream()
    ImageIO.write(img, "png", baos)
    baos.toByteArray

  def spec = suite("ImageTransformer (DL4J — ND4J)")(
    test("transform raw image bytes returns non-empty result") {
      val pipeline = ImagePipeline(ImageTransformDef.Resize(10, 10))
      val transformer = ImageTransformer(pipeline)
      val bytes = createTestImageBytes(20, 20)
      val result = transformer.transform(bytes)
      assertTrue(result.isSuccess, result.get.nonEmpty)
    },
    test("resize changes image dimensions") {
      val pipeline = ImagePipeline(ImageTransformDef.Resize(5, 10))
      val transformer = ImageTransformer(pipeline)
      val bytes = createTestImageBytes(40, 30)
      val result = transformer.transform(bytes).get
      assertTrue(result.length == 5, result.head.length == 30)
    },
    test("normalize transform does not crash") {
      val pipeline = ImagePipeline(
        ImageTransformDef.Resize(10, 10),
        ImageTransformDef.Normalize(Array(0.5f, 0.5f, 0.5f), Array(0.25f, 0.25f, 0.25f))
      )
      val transformer = ImageTransformer(pipeline)
      val bytes = createTestImageBytes(20, 20)
      val result = transformer.transform(bytes)
      assertTrue(result.isSuccess)
    },
    test("grayscale image produces single-channel output") {
      val pipeline = ImagePipeline(ImageTransformDef.Resize(28, 28))
      val transformer = ImageTransformer(pipeline)
      val bytes = createTestImageBytes(14, 14, gray = true)
      val result = transformer.transform(bytes).get
      assertTrue(result.length == 28, result.head.length == 28)
    },
    test("grayscale with single-channel normalize") {
      val pipeline = ImagePipeline(
        ImageTransformDef.Resize(28, 28),
        ImageTransformDef.Normalize(Array(0.1307f), Array(0.3081f))
      )
      val transformer = ImageTransformer(pipeline)
      val bytes = createTestImageBytes(14, 14, gray = true)
      val result = transformer.transform(bytes)
      assertTrue(result.isSuccess, result.get.length == 28, result.get.head.length == 28)
    }
  )
