package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import java.awt.image.BufferedImage

object ImageTransformerSpec extends ZIOSpecDefault:

  private def createTestImageBytes(width: Int, height: Int): Array[Byte] =
    val img = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
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
    }
  )
