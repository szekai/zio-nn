package zio.nn.storch

import torch.*
import torch.DType.*
import zio.nn.{ImagePipeline, ImageTransformDef}
import zio.{Task, ZIO}
import scala.util.Try

/** Image preprocessing utilities for the storch (bytedeco/storch) backend.
  *
  * Loads image files and converts them to [[Tensor[Float32]]] in CHW format
  * (channels × height × width — standard PyTorch layout) with pixel values
  * normalized to [0, 1]. Supports resize, normalize, and center-crop transforms
  * via [[ImagePipeline]].
  *
  * {{{
  *   val t = ImageTransformer.loadImage("photo.jpg").get
  *   val pipeline = ImagePipeline(Resize(224, 224), Normalize(mean, std))
  *   val transformed = ImageTransformer.applyPipeline(t, pipeline).get
  * }}}
  */
object ImageTransformer:

  /** Load an image from file, returning a CHW tensor with values in [0, 1].
    *
    * @param path
    *   filesystem path to the image file (JPEG / PNG / etc.)
    * @return
    *   [[Tensor[Float32]]] with shape `(channels, height, width)`
    */
  def loadImage(path: String): Try[Tensor[Float32]] = Try {
    import javax.imageio.ImageIO
    import java.io.File

    val buf = ImageIO.read(new File(path))
    if buf == null then throw new java.io.IOException(s"Failed to read image: $path")

    val h  = buf.getHeight
    val w  = buf.getWidth
    val ch = channelCount(buf)
    val px = bufferedImageToCHW(buf, h, w, ch)
    torch.Tensor(px).reshape(ch, h, w)
  }

  /** Apply a pipeline of image transforms to a tensor.
    *
    * Each transform in `pipeline.transforms` is applied sequentially to the
    * image tensor.
    */
  def applyPipeline(
      image: Tensor[Float32],
      pipeline: ImagePipeline
  ): Try[Tensor[Float32]] = Try {
    var result = image
    for t <- pipeline.transforms do
      result = applyTransformUnsafe(result, t)
    result
  }

  /** Apply a single image transform to a tensor. */
  def applyTransform(
      image: Tensor[Float32],
      transform: ImageTransformDef
  ): Try[Tensor[Float32]] =
    Try(applyTransformUnsafe(image, transform))

  // ── ZIO variants ──────────────────────────────────────────────────────────

  /** ZIO variant of [[loadImage]]. */
  def loadImageZ(path: String): Task[Tensor[Float32]] =
    ZIO.fromTry(loadImage(path))

  /** ZIO variant of [[applyPipeline]]. */
  def applyPipelineZ(
      image: Tensor[Float32],
      pipeline: ImagePipeline
  ): Task[Tensor[Float32]] =
    ZIO.fromTry(applyPipeline(image, pipeline))

  /** ZIO variant of [[applyTransform]]. */
  def applyTransformZ(
      image: Tensor[Float32],
      transform: ImageTransformDef
  ): Task[Tensor[Float32]] =
    ZIO.fromTry(applyTransform(image, transform))

  // ── Internal dispatch ─────────────────────────────────────────────────────

  private def applyTransformUnsafe(
      image: Tensor[Float32],
      transform: ImageTransformDef
  ): Tensor[Float32] = transform match
    case ImageTransformDef.Resize(h, w)    => resizeTensor(image, h, w)
    case ImageTransformDef.Normalize(m, s) => normalizeTensor(image, m, s)
    case ImageTransformDef.CenterCrop(h, w) => centerCropTensor(image, h, w)

  // ── BufferedImage → CHW float array ───────────────────────────────────────

  private def channelCount(buf: java.awt.image.BufferedImage): Int =
    buf.getType match
      case java.awt.image.BufferedImage.TYPE_BYTE_GRAY => 1
      case _ => if buf.getColorModel.hasAlpha then 4 else 3

  /** Convert [[java.awt.image.BufferedImage]] to a flat float array in CHW
    * layout (channels × height × width). Pixel values are normalized to [0, 1].
    */
  private def bufferedImageToCHW(
      buf: java.awt.image.BufferedImage,
      h: Int,
      w: Int,
      ch: Int
  ): Array[Float] =
    val pixels = new Array[Float](ch * h * w)
    if ch == 1 then
      val data = buf.getData.getPixels(0, 0, w, h, null: Array[Float])
      var i = 0
      val n = h * w
      while i < n do
        pixels(i) = data(i) / 255f
        i += 1
    else
      val rgb   = buf.getRGB(0, 0, w, h, null, 0, w)
      val nCh   = if ch == 4 then 4 else 3
      val total = h * w
      for c <- 0 until nCh do
        var dst = c * total
        var src = 0
        while src < total do
          val pixel = rgb(src)
          pixels(dst) = (c match
            case 0 => (pixel >> 16) & 0xFF
            case 1 => (pixel >> 8) & 0xFF
            case 2 => pixel & 0xFF
            case 3 => (pixel >> 24) & 0xFF
          ).toFloat / 255f
          dst += 1
          src += 1
    pixels

  // ── Transform implementations ─────────────────────────────────────────────

  /** Nearest-neighbour resize on CHW tensor data. */
  private def resizeTensor(
      t: Tensor[Float32],
      tH: Int,
      tW: Int
  ): Tensor[Float32] =
    val s   = t.shape
    val ch  = s(0).toInt
    val hIn = s(1).toInt
    val wIn = s(2).toInt

    val flat = t.toArray

    val out = new Array[Float](ch * tH * tW)
    for c <- 0 until ch do
      for y <- 0 until tH do
        val sy = y * hIn / tH
        for x <- 0 until tW do
          val sx = x * wIn / tW
          out(c * tH * tW + y * tW + x) =
            flat(c * hIn * wIn + sy * wIn + sx)

    torch.Tensor(out).reshape(ch, tH, tW)

  /** Per-channel (tensor - mean) / std on CHW tensor data. */
  private def normalizeTensor(
      t: Tensor[Float32],
      mean: Array[Float],
      std: Array[Float]
  ): Tensor[Float32] =
    val s  = t.shape
    val ch = s(0).toInt
    val h  = s(1).toInt
    val w  = s(2).toInt

    val flat = t.toArray

    for c <- 0 until ch do
      val m   = if c < mean.length then mean(c) else mean.last
      val st  = if c < std.length then std(c) else std.last
      var idx = c * h * w
      val end = idx + h * w
      while idx < end do
        flat(idx) = (flat(idx) - m) / st
        idx += 1

    torch.Tensor(flat).reshape(ch, h, w)

  /** Centre-crop a CHW tensor to target (height, width). */
  private def centerCropTensor(
      t: Tensor[Float32],
      tH: Int,
      tW: Int
  ): Tensor[Float32] =
    val s   = t.shape
    val ch  = s(0).toInt
    val hIn = s(1).toInt
    val wIn = s(2).toInt
    val yOff = (hIn - tH) / 2
    val xOff = (wIn - tW) / 2

    val flat = t.toArray

    val out = new Array[Float](ch * tH * tW)
    for c <- 0 until ch do
      val srcBase = c * hIn * wIn
      val dstBase = c * tH * tW
      for y <- 0 until tH do
        val srcRow = (yOff + y) * wIn + xOff
        val dstRow = y * tW
        System.arraycopy(flat, srcBase + srcRow, out, dstBase + dstRow, tW)

    torch.Tensor(out).reshape(ch, tH, tW)

end ImageTransformer
