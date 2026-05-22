package zio.nn

/** DSL sugar — cleaner syntax for architecture definitions.
  *
  * Before (verbose):
  * {{{
  *   ModelDef.Sequential(SequentialDef(7, List(
  *     LayerDef.LSTM(7, 64, ActivationFn.Tanh),
  *     LayerDef.Dense(64, 32, ActivationFn.ReLU),
  *     LayerDef.Output(32, 1, LossFn.MSE)
  *   )))
  * }}}
  *
  * After (DSL):
  * {{{
  *   Sequential(7)(LSTM(64, Tanh), Dense(32, ReLU), Output(1, MSE))
  * }}}
  */
object dsl:

  // ═══ Layer constructors (input size auto-inferred) ═══

  def LSTM(nOut: Int, activation: ActivationFn = ActivationFn.Tanh, dropout: Double = 0.0): LayerSpec =
    LayerSpec.LSTM(nOut, activation, dropout)

  def Dense(nOut: Int, activation: ActivationFn = ActivationFn.ReLU): LayerSpec =
    LayerSpec.Dense(nOut, activation)

  def Output(nOut: Int, loss: LossFn = LossFn.MSE, activation: ActivationFn = ActivationFn.Identity): LayerSpec =
    LayerSpec.Output(nOut, loss, activation)

  def BatchNorm: LayerSpec = LayerSpec.BatchNorm
  def Dropout(rate: Double): LayerSpec = LayerSpec.Dropout(rate)

  // ═══ Activation / Loss / Optimizer shortcuts ═══
  val Tanh    = ActivationFn.Tanh
  val ReLU    = ActivationFn.ReLU
  val Sigmoid = ActivationFn.Sigmoid
  val Softmax = ActivationFn.Softmax
  val MSE  = LossFn.MSE
  val MAE  = LossFn.MAE
  def Adam(lr: Double = 0.001)  = OptimizerDef.Adam(lr)
  def SGD(lr: Double = 0.01)    = OptimizerDef.SGD(lr)
  def RMSprop(lr: Double = 0.001) = OptimizerDef.RMSprop(lr)

  // ═══ Layer spec (intermediate representation) ═══

  enum LayerSpec:
    case LSTM(nOut: Int, activation: ActivationFn, dropout: Double)
    case Dense(nOut: Int, activation: ActivationFn)
    case Output(nOut: Int, loss: LossFn, activation: ActivationFn)
    case BatchNorm
    case Dropout(rate: Double)

    def resolve(nIn: Int): LayerDef = this match
      case LSTM(nOut, act, drop) => LayerDef.LSTM(nIn, nOut, act, drop)
      case Dense(nOut, act)      => LayerDef.Dense(nIn, nOut, act)
      case Output(nOut, loss, act) => LayerDef.Output(nIn, nOut, loss, act)
      case LayerSpec.BatchNorm => LayerDef.BatchNorm(nIn)
      case Dropout(rate)         => LayerDef.Dropout(rate)

    def outputSize: Int = this match
      case LSTM(nOut, _, _)  => nOut
      case Dense(nOut, _)    => nOut
      case Output(nOut, _, _)=> nOut
      case LayerSpec.BatchNorm => -1
      case LayerSpec.Dropout(_) => -1

  // ═══ Sequential builder ═══

  class SequentialBuilder private[dsl] (inputSize: Int, layers: List[LayerSpec], optimizer: OptimizerDef, seed: Long):
    def apply(more: LayerSpec*): SequentialBuilder =
      new SequentialBuilder(inputSize, layers ++ more, optimizer, seed)

    def withOptimizer(opt: OptimizerDef): SequentialBuilder =
      new SequentialBuilder(inputSize, layers, opt, seed)

    def withSeed(s: Long): SequentialBuilder =
      new SequentialBuilder(inputSize, layers, optimizer, s)

    def build: ModelDef =
      val resolved = layers.foldLeft((inputSize, List.empty[LayerDef])) { case ((nIn, acc), spec) =>
        val layer = spec.resolve(nIn)
        val nextN = spec.outputSize match
          case -1 => nIn
          case n  => n
        (nextN, acc :+ layer)
      }
      ModelDef.Sequential(SequentialDef(inputSize, resolved._2, optimizer, seed))


  def Sequential(inputSize: Int): SequentialBuilder =
    new SequentialBuilder(inputSize, Nil, OptimizerDef.Adam(), 42L)
