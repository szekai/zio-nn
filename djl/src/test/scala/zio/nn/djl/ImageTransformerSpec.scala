package zio.nn.djl

import zio.*
import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object ImageTransformerSpec extends ZIOSpecDefault:

  private val ndmLayer: ULayer[ai.djl.ndarray.NDManager] =
    ZLayer.scoped(
      ZIO.acquireRelease(
        ZIO.attempt(ai.djl.ndarray.NDManager.newBaseManager()).orDie
      )(ndm => ZIO.attempt(ndm.close()).orDie)
    )

  private def createWhitePngBytes(width: Int, height: Int): Array[Byte] =
    import java.awt.image.BufferedImage
    import java.io.ByteArrayOutputStream
    import javax.imageio.ImageIO
    val bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g = bi.getGraphics
    g.setColor(java.awt.Color.WHITE)
    g.fillRect(0, 0, width, height)
    g.dispose()
    val baos = new ByteArrayOutputStream()
    ImageIO.write(bi, "png", baos)
    baos.toByteArray

  def spec = suite("ImageTransformer (DJL)")(
    test("construction with empty pipeline") {
      val pipeline = ImagePipeline()
      val transformer = ImageTransformer(pipeline)
      assertTrue(transformer != null)
    },
    test("construction with resize transform") {
      val pipeline = ImagePipeline(ImageTransformDef.Resize(224, 224))
      val transformer = ImageTransformer(pipeline)
      assertTrue(transformer != null)
    },
    test("construction with all transform types") {
      val pipeline = ImagePipeline(
        ImageTransformDef.Resize(224, 224),
        ImageTransformDef.Normalize(Array(0.5f, 0.5f, 0.5f), Array(0.25f, 0.25f, 0.25f)),
        ImageTransformDef.CenterCrop(200, 200)
      )
      val transformer = ImageTransformer(pipeline)
      assertTrue(transformer != null)
    },
    test("transform after resize succeeds") {
      ZIO.service[ai.djl.ndarray.NDManager].flatMap { implicit ndm =>
        val pipeline = ImagePipeline(ImageTransformDef.Resize(1, 1))
        val transformer = ImageTransformer(pipeline)
        val imageBytes = createWhitePngBytes(2, 2)
        val result = transformer.transform(imageBytes)
        ZIO.attempt(assertTrue(result.isSuccess))
      }
    },
    test("transform with Normalize preserves output non-empty") {
      ZIO.service[ai.djl.ndarray.NDManager].flatMap { implicit ndm =>
        val pipeline = ImagePipeline(
          ImageTransformDef.Resize(2, 2),
          ImageTransformDef.Normalize(Array(0.5f, 0.5f, 0.5f), Array(0.5f, 0.5f, 0.5f))
        )
        val transformer = ImageTransformer(pipeline)
        val imageBytes = createWhitePngBytes(4, 4)
        val result = transformer.transform(imageBytes)
        ZIO.attempt(assertTrue(result.isSuccess, result.get.nonEmpty))
      }
    },
    test("transform with CenterCrop produces expected dimensions") {
      ZIO.service[ai.djl.ndarray.NDManager].flatMap { implicit ndm =>
        val pipeline = ImagePipeline(
          ImageTransformDef.CenterCrop(2, 2)
        )
        val transformer = ImageTransformer(pipeline)
        val imageBytes = createWhitePngBytes(4, 4)
        val result = transformer.transform(imageBytes)
        ZIO.attempt(assertTrue(result.isSuccess, result.get.length == 6))
      }
    }
  ).provideLayer(ndmLayer)
