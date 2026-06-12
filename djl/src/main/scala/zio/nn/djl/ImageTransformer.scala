package zio.nn.djl

import ai.djl.modality.cv.Image
import ai.djl.modality.cv.util.NDImageUtils
import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.ndarray.types.DataType
import zio.nn.{ImagePipeline, ImageTransformDef}
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import scala.util.Try

/** Applies a [[zio.nn.ImagePipeline]] to raw image bytes using DJL's CV transforms.
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
    * @param manager    implicit NDManager for NDArray lifecycle
    * @return (height, width, channels) × feature rows as Array[Array[Float]]
    */
  def transform(imageBytes: Array[Byte])(using manager: NDManager): Try[Array[Array[Float]]] = Try {
    val bis = new ByteArrayInputStream(imageBytes)
    val buf = ImageIO.read(bis)
    val img = ai.djl.modality.cv.ImageFactory.getInstance().fromImage(buf)
    var arr: NDArray = img.toNDArray(manager)

    for t <- pipeline.transforms do
      arr = t match
        case ImageTransformDef.Resize(h, w) =>
          NDImageUtils.resize(arr, w, h)
        case ImageTransformDef.Normalize(mean, std) =>
          val meanArr = manager.create(mean)
          val stdArr  = manager.create(std)
          arr.sub(meanArr).div(stdArr)
        case ImageTransformDef.CenterCrop(h, w) =>
          NDImageUtils.centerCrop(arr, h, w)

    val arrFloat = arr.toType(DataType.FLOAT32, false)
    val shape = arrFloat.getShape
    val rows  = if shape.dimension() > 0 then shape.get(0).toInt else 1
    val cols  = if shape.dimension() > 1 then shape.get(1).toInt else shape.get(shape.dimension() - 1).toInt max 1
    val flat  = arrFloat.toFloatArray
    flat.grouped(cols).toArray
  }

object ImageTransformer:
  def apply(pipeline: ImagePipeline): ImageTransformer = new ImageTransformer(pipeline)
