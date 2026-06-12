package zio.nn.dl4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import zio.nn.{ImagePipeline, ImageTransformDef}
import scala.util.Try

/** Applies a [[zio.nn.ImagePipeline]] to raw image bytes using ND4J operations.
  *
  * {{{
  *   val pipeline = ImagePipeline(Resize(224, 224), Normalize(mean, std))
  *   val transformer = ImageTransformer(pipeline)
  *   val result: Try[Array[Array[Float]]] = transformer.transform(imageBytes)
  * }}}
  */
class ImageTransformer(pipeline: ImagePipeline):

  /** Transform raw image bytes into a float array ready for model prediction.
    *
    * @param imageBytes raw image file bytes (JPEG/PNG)
    * @return (height × width × channels) feature rows as Array[Array[Float]]
    */
  def transform(imageBytes: Array[Byte]): Try[Array[Array[Float]]] = Try {
    import javax.imageio.ImageIO
    import java.io.ByteArrayInputStream

    val buf  = ImageIO.read(new ByteArrayInputStream(imageBytes))
    val hIn  = buf.getHeight
    val wIn  = buf.getWidth
    val ch   = buf.getType match
                case java.awt.image.BufferedImage.TYPE_BYTE_GRAY => 1
                case _ => if buf.getColorModel.hasAlpha then 4 else 3
    var arr  = rasterToINDArray(buf, hIn, wIn, ch)

    for t <- pipeline.transforms do
      arr = t match
        case ImageTransformDef.Resize(h, w) =>
          resizeINDArray(arr, h, w, hIn, wIn)
        case ImageTransformDef.Normalize(mean, std) =>
          val ch = mean.length
          val expandedMean = Array.tabulate(arr.columns().toInt)(i => mean(i % ch))
          val expandedStd  = Array.tabulate(arr.columns().toInt)(i => std(i % ch))
          arr.sub(Nd4j.create(expandedMean)).div(Nd4j.create(expandedStd))
        case ImageTransformDef.CenterCrop(h, w) =>
          centerCropINDArray(arr, h, w)

    val rows = arr.rows().toInt
    val cols = arr.columns().toInt
    arr.ravel().toFloatVector.grouped(cols).toArray
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  private def rasterToINDArray(buf: java.awt.image.BufferedImage, h: Int, w: Int, ch: Int): INDArray = {
    if buf.getType == java.awt.image.BufferedImage.TYPE_BYTE_GRAY then
      val data = buf.getData.getPixels(0, 0, w, h, null: Array[Float])
      Nd4j.create(data).reshape(h, w)
    else
      val pixels = buf.getRGB(0, 0, w, h, null, 0, w)
      val data   = new Array[Float](h * w * ch)
      var idx    = 0
      for (y <- 0 until h; x <- 0 until w) {
        val rgb   = pixels(y * w + x)
        val r     = ((rgb >> 16) & 0xFF).toFloat / 255f
        val g     = ((rgb >> 8) & 0xFF).toFloat / 255f
        val b     = (rgb & 0xFF).toFloat / 255f
        if (ch == 4) {
          val a = ((rgb >> 24) & 0xFF).toFloat / 255f
          data(idx) = r; data(idx + 1) = g; data(idx + 2) = b; data(idx + 3) = a
          idx += 4
        } else {
          data(idx) = r; data(idx + 1) = g; data(idx + 2) = b
          idx += 3
        }
      }
      Nd4j.create(data).reshape(h, w * ch)
  }

  private def resizeINDArray(arr: INDArray, tH: Int, tW: Int, hIn: Int, wIn: Int): INDArray = {
    val ch   = arr.columns().toInt / wIn
    val flat = arr.ravel().toFloatVector
    val out  = new Array[Float](tH * tW * ch)
    for (y <- 0 until tH; x <- 0 until tW) {
      val sy = (y * hIn) / tH
      val sx = (x * wIn) / tW
      for (c <- 0 until ch) {
        out((y * tW + x) * ch + c) = flat((sy * wIn + sx) * ch + c)
      }
    }
    Nd4j.create(out).reshape(tH, tW * ch)
  }

  private def centerCropINDArray(arr: INDArray, tH: Int, tW: Int): INDArray = {
    val hIn  = arr.rows().toInt
    val wIn  = arr.columns().toInt
    val ch   = wIn / hIn
    val flat = arr.ravel().toFloatVector
    val yOff = (hIn - tH) / 2
    val xOff = (wIn / ch - tW) / 2
    val out  = new Array[Float](tH * tW * ch)
    var idx  = 0
    for (y <- 0 until tH; x <- 0 until tW; c <- 0 until ch) {
      out(idx) = flat(((yOff + y) * (wIn / ch) + (xOff + x)) * ch + c)
      idx += 1
    }
    Nd4j.create(out).reshape(tH, tW * ch)
  }

object ImageTransformer:
  def apply(pipeline: ImagePipeline): ImageTransformer = new ImageTransformer(pipeline)
