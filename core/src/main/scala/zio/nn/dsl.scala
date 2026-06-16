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
  def Conv2D(filters: Int, kernel: (Int, Int) = (3, 3), stride: (Int, Int) = (1, 1), activation: ActivationFn = ActivationFn.ReLU): LayerSpec =
    LayerSpec.Conv2D(filters, kernel, stride, activation)
  def MaxPool2D(poolSize: (Int, Int) = (2, 2)): LayerSpec = LayerSpec.MaxPool2D(poolSize)
  def Flatten: LayerSpec = LayerSpec.Flatten

  /** Create a randomly-initialized embedding layer.
    *
    * @param vocabSize    number of unique tokens
    * @param embeddingDim output vector dimension
    */
  def Embedding(vocabSize: Int, embeddingDim: Int): LayerSpec =
    LayerSpec.Embedding(vocabSize, embeddingDim, None)

  /** Create an embedding layer with pre-trained weights.
    * Called by `Word2VecModel.toEmbeddingLayer()` in the embeddings module.
    */
  def Embedding(vocabSize: Int, embeddingDim: Int, weights: EmbeddingWeights): LayerSpec =
    LayerSpec.Embedding(vocabSize, embeddingDim, Some(weights))

  /** Gated Recurrent Unit — simpler alternative to LSTM. */
  def GRU(nOut: Int, activation: ActivationFn = ActivationFn.Tanh, dropout: Double = 0.0): LayerSpec =
    LayerSpec.GRU(nOut, activation, dropout)

  /** Bidirectional wrapper — doubles output dimension via concatenation. */
  def BiDirectional(inner: LayerSpec): LayerSpec =
    LayerSpec.BiDirectional(inner)

  /** Multi-head self-attention (core of Transformer architectures). */
  def MultiHeadAttention(embeddingDim: Int, numHeads: Int, dropout: Double = 0.0): LayerSpec =
    LayerSpec.MultiHeadAttention(embeddingDim, numHeads, dropout)

  /** Layer normalization — normalizes across feature dimension.
    * Required for Transformer architectures.
    */
  val LayerNorm: LayerSpec = LayerSpec.LayerNorm

  /** Transformer encoder block — full encoder with self-attention + FFN.
    *
    * @param dim       model dimension (embedding size)
    * @param numHeads  number of attention heads
    * @param ffDim     feed-forward hidden size (typically 4 × dim)
    * @param numLayers number of stacked encoder layers (default: 1)
    * @param dropout   dropout probability (default: 0.1)
    */
  def TransformerEncoder(dim: Int, numHeads: Int, ffDim: Int, numLayers: Int = 1, dropout: Double = 0.1): LayerSpec =
    LayerSpec.TransformerEncoder(dim, numHeads, ffDim, numLayers, dropout)

  // ═══ Activation / Loss / Optimizer shortcuts ═══
  val Tanh    = ActivationFn.Tanh
  val ReLU    = ActivationFn.ReLU
  val Sigmoid = ActivationFn.Sigmoid
  val Softmax = ActivationFn.Softmax
  def LeakyReLU(alpha: Double = 0.01): ActivationFn = ActivationFn.LeakyReLU(alpha)
  val MSE  = LossFn.MSE
  val MAE  = LossFn.MAE
  def BinaryCrossEntropy(epsilon: Double = 1e-15): LossFn = LossFn.BinaryCrossEntropy(epsilon)
  def CategoricalCrossEntropy(epsilon: Double = 1e-15): LossFn = LossFn.CategoricalCrossEntropy(epsilon)
  def Huber(delta: Double = 1.0): LossFn = LossFn.Huber(delta)
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
    case Conv2D(filters: Int, kernel: (Int, Int), stride: (Int, Int), activation: ActivationFn)
    case MaxPool2D(poolSize: (Int, Int))
    case Flatten
    case Embedding(vocabSize: Int, embeddingDim: Int, pretrained: Option[EmbeddingWeights] = None)
    case GRU(nOut: Int, activation: ActivationFn, dropout: Double)
    case BiDirectional(inner: LayerSpec)
    case MultiHeadAttention(embeddingDim: Int, numHeads: Int, dropout: Double)
    case LayerNorm
    case TransformerEncoder(dim: Int, numHeads: Int, ffDim: Int, numLayers: Int, dropout: Double)

    def resolve(nIn: Int): AnyLayer = this match
      case LayerSpec.LSTM(nOut, act, drop) => AnyLayer.Standard(LayerDef.LSTM(nIn, nOut, act, drop))
      case LayerSpec.Dense(nOut, act)      => AnyLayer.Standard(LayerDef.Dense(nIn, nOut, act))
      case LayerSpec.Output(nOut, loss, act) => AnyLayer.Standard(LayerDef.Output(nIn, nOut, loss, act))
      case LayerSpec.BatchNorm             => AnyLayer.Standard(LayerDef.BatchNorm(nIn))
      case LayerSpec.Dropout(rate)         => AnyLayer.Standard(LayerDef.Dropout(rate))
      case LayerSpec.Conv2D(filters, kernel, stride, act) => AnyLayer.Standard(LayerDef.Conv2D(nIn, filters, kernel, stride, act))
      case LayerSpec.MaxPool2D(poolSize)   => AnyLayer.Standard(LayerDef.MaxPool2D(poolSize))
      case LayerSpec.Flatten               => AnyLayer.Standard(LayerDef.Flatten)
      case LayerSpec.Embedding(vocabSize, embeddingDim, pretrained) =>
        AnyLayer.Standard(LayerDef.Embedding(vocabSize, embeddingDim, pretrained))
      case LayerSpec.GRU(nOut, act, drop) => AnyLayer.Advanced(AdvancedLayerDef.GRU(nIn, nOut, act, drop))
      case LayerSpec.BiDirectional(inner) =>
        val resolved = inner.resolve(nIn) match
          case AnyLayer.Standard(LayerDef.LSTM(_, nOut, act, drop)) =>
            (BidirectionalKind.LSTM, nOut, act, drop)
          case _ => throw IllegalArgumentException(s"BiDirectional inner layer must be LSTM")
        AnyLayer.Advanced(AdvancedLayerDef.BiDirectional(resolved._1, nIn, resolved._2, resolved._3, resolved._4))
      case LayerSpec.MultiHeadAttention(embeddingDim, numHeads, dropout) =>
        AnyLayer.Advanced(AdvancedLayerDef.MultiHeadAttention(embeddingDim, numHeads, dropout))
      case LayerSpec.LayerNorm =>
        AnyLayer.Standard(LayerDef.LayerNorm(nIn))
      case LayerSpec.TransformerEncoder(dim, numHeads, ffDim, numLayers, dropout) =>
        AnyLayer.Advanced(AdvancedLayerDef.TransformerEncoder(dim, numHeads, ffDim, numLayers, dropout))

    def outputSize: Int = this match
      case LayerSpec.LSTM(nOut, _, _)   => nOut
      case LayerSpec.Dense(nOut, _)     => nOut
      case LayerSpec.Output(nOut, _, _) => nOut
      case LayerSpec.BatchNorm          => -1
      case LayerSpec.Dropout(_)         => -1
      case LayerSpec.Conv2D(_, _, _, _) => -1
      case LayerSpec.MaxPool2D(_)       => -1
      case LayerSpec.Flatten            => -1
      case LayerSpec.Embedding(_, embeddingDim, _) => embeddingDim
      case LayerSpec.GRU(nOut, _, _)           => nOut
      case LayerSpec.BiDirectional(inner)      => inner.outputSize * 2
      case LayerSpec.MultiHeadAttention(embeddingDim, _, _) => embeddingDim
      case LayerSpec.LayerNorm          => -1
      case LayerSpec.TransformerEncoder(dim, _, _, _, _) => dim

  // ═══ Sequential builder ═══

  class SequentialBuilder private[dsl] (inputSize: Int, layers: List[LayerSpec], optimizer: OptimizerDef, seed: Long, convInput: Option[ConvInput] = None):
    def apply(more: LayerSpec*): SequentialBuilder =
      new SequentialBuilder(inputSize, layers ++ more, optimizer, seed, convInput)

    def withOptimizer(opt: OptimizerDef): SequentialBuilder =
      new SequentialBuilder(inputSize, layers, opt, seed, convInput)

    def withSeed(s: Long): SequentialBuilder =
      new SequentialBuilder(inputSize, layers, optimizer, s, convInput)

    /** Set spatial input dimensions for Conv2D models.
      * Required when the first layer is Conv2D — tells the backend
      * to configure `InputType.convolutional(height, width, channels)`.
      */
    def withConvInput(height: Int, width: Int, channels: Int): SequentialBuilder =
      new SequentialBuilder(inputSize, layers, optimizer, seed, Some(ConvInput(height, width, channels)))

    def build: ModelDef =
      val resolved = layers.foldLeft((inputSize, List.empty[AnyLayer])) { case ((nIn, acc), spec) =>
        val anyLayer = spec.resolve(nIn)
        val nextN = spec.outputSize match
          case -1 => nIn
          case n  => n
        (nextN, acc :+ anyLayer)
      }
      ModelDef.Sequential(SequentialDef(inputSize, resolved._2, optimizer, seed, convInput))


  def Sequential(inputSize: Int): SequentialBuilder =
    new SequentialBuilder(inputSize, Nil, OptimizerDef.Adam(), 42L)
