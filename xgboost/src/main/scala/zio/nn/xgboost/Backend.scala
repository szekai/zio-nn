package zio.nn.xgboost

import zio.nn.*
import scala.util.{Failure, Success, Try}

/** XGBoost backend — gradient-boosted trees via xgboost4j.
  *
  * Mirrors the zio-nn backend contract (duck-typed, like djl/dl4j/storch) so a
  * consumer can swap `zio.nn.djl.ZModel` → `zio.nn.xgboost.ZModel` and keep the
  * same train/predict/save/load loop. Trees have no recurrence or convolution,
  * so only a `Sequential` of `Dense`/`Output` layers is supported — anything
  * else is rejected at compile time with a clear error.
  */
object Backend:

  final case class XgbConfig(inputSize: Int, outputSize: Int)

  def compile(model: ModelDef): Try[XgbConfig] = model match
    case ModelDef.Sequential(seq) =>
      val unsupported = seq.layers.collect {
        case AnyLayer.Advanced(l)                       => l.toString
        case AnyLayer.Standard(l) if !isSupported(l)    => l.toString
      }
      if unsupported.nonEmpty then
        Failure(new UnsupportedOperationException(
          s"XGBoost supports only Dense/Output layers; unsupported: ${unsupported.mkString(", ")}"))
      else
        val outputSize = seq.layers.collect {
          case AnyLayer.Standard(LayerDef.Output(_, nOut, _, _)) => nOut
        }.lastOption
          .orElse(seq.layers.collect {
            case AnyLayer.Standard(LayerDef.Dense(_, nOut, _)) => nOut
          }.lastOption)
          .getOrElse(1)
        Success(XgbConfig(inputSize = seq.inputSize, outputSize = outputSize))
    case ModelDef.Functional(_) =>
      Failure(new UnsupportedOperationException(
        "XGBoost supports only Sequential architectures (functional/DAG graphs are not tree-mappable)"))

  private def isSupported(l: LayerDef): Boolean = l match
    case _: LayerDef.Dense  => true
    case _: LayerDef.Output => true
    case _                  => false
