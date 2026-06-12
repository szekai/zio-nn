package zio.nn

import zio.{UIO, ZIO}

/** Framework-agnostic neural network architecture definition.
  *
  * Define your model structure once as pure data, then compile to any backend:
  * {{{
  *   val arch = Sequential(7,
  *     LSTM(7 -> 64, Tanh),
  *     Dense(64 -> 32, ReLU),
  *     Output(32 -> 1, MSE))
  *
  *   val block: Block = BackendDJL.compile(arch)     // DJL
  *   val net: MultiLayerNetwork = BackendDL4J.compile(arch) // DL4J
  * }}}
  *
  * == Escape Hatch ==
  * This DSL covers ~80% of use cases. For custom layers, multi-GPU, or
  * transfer learning, bypass the DSL and use the raw framework API directly
  * via `.underlying` on Model, Predictor, and Trainer.
  */
object ModelArchitecture

// ═══════════════════════════════════════════════════════════
//  Layer Definitions
// ═══════════════════════════════════════════════════════════

enum LayerDef:
  case LSTM(nIn: Int, nOut: Int, activation: ActivationFn = ActivationFn.Tanh, dropout: Double = 0.0)
  case Dense(nIn: Int, nOut: Int, activation: ActivationFn = ActivationFn.ReLU)
  case Output(nIn: Int, nOut: Int, loss: LossFn = LossFn.MSE, activation: ActivationFn = ActivationFn.Identity)
  case BatchNorm(nIn: Int)
  case Dropout(rate: Double)
  case Conv2D(nIn: Int, filters: Int, kernel: (Int, Int) = (3, 3), stride: (Int, Int) = (1, 1), activation: ActivationFn = ActivationFn.ReLU)
  case MaxPool2D(poolSize: (Int, Int) = (2, 2))
  case Flatten

  /** Layer normalization — normalizes across feature dimension.
    * Required building block for Transformer architectures.
    */
  case LayerNorm(nIn: Int)

  /** Maps discrete token IDs to dense vector representations.
    * When used as first layer, model input is `Array[Array[Int]]` (token indices).
    * Set `pretrained = Some(EmbeddingWeights(...))` to initialize from pre-trained vectors.
    *
    * @param vocabSize     number of unique tokens in vocabulary
    * @param embeddingDim  dimension of each embedding vector (e.g. 300 for Google News)
    * @param pretrained    pre-trained weights to initialize the embedding table (None = random init)
    */
  case Embedding(vocabSize: Int, embeddingDim: Int, pretrained: Option[EmbeddingWeights] = None)

/** Pre-trained embedding vectors — framework-free bridge type. */
case class EmbeddingWeights(vocabulary: Map[String, Int], vectors: Array[Array[Float]])

/** Advanced layers not configurable via HOCON (programmatic-only DSL). */
enum AdvancedLayerDef:
  case GRU(nIn: Int, nOut: Int, activation: ActivationFn = ActivationFn.Tanh, dropout: Double = 0.0)
  case BiDirectional(kind: BidirectionalKind, nIn: Int, nOut: Int, activation: ActivationFn, dropout: Double)
  case MultiHeadAttention(embeddingDim: Int, numHeads: Int, dropout: Double = 0.0)

  /** Transformer encoder block — stacks of self-attention + feed-forward.
    * Wraps MultiHeadAttention, residual connections, LayerNorm, and FFN
    * into a single block. DJL compiles natively; DL4J users should use DJL
    * backend or manually compose via FunctionalDef.
    *
    * @param dim       model dimension (embedding size)
    * @param numHeads  number of attention heads
    * @param ffDim     feed-forward hidden dimension (typically 4 × dim)
    * @param numLayers number of stacked encoder layers
    * @param dropout   dropout probability
    */
  case TransformerEncoder(dim: Int, numHeads: Int, ffDim: Int, numLayers: Int = 1, dropout: Double = 0.1)

enum BidirectionalKind:
  case LSTM, GRU

// ═══════════════════════════════════════════════════════════
//  Activation Functions
// ═══════════════════════════════════════════════════════════

enum ActivationFn:
  case Tanh, ReLU, Sigmoid, Softmax, Identity, LeakyReLU

// ═══════════════════════════════════════════════════════════
//  Loss Functions
// ═══════════════════════════════════════════════════════════

enum LossFn:
  case MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy, Huber

// ═══════════════════════════════════════════════════════════
//  Optimizers
// ═══════════════════════════════════════════════════════════

enum OptimizerDef:
  case Adam(learningRate: Double = 0.001)
  case SGD(learningRate: Double = 0.01)
  case RMSprop(learningRate: Double = 0.001)

// ═══════════════════════════════════════════════════════════
//  Model Definitions
// ═══════════════════════════════════════════════════════════

/** Wrapper to include both standard and advanced layers in model definitions. */
enum AnyLayer:
  case Standard(layer: LayerDef)
  case Advanced(layer: AdvancedLayerDef)

/** Sequential model: layers stacked in order. Covers 80% of use cases.
  *
  * @param inputSize  input dimension (features for flat, channels for Conv2D)
  * @param layers     ordered list of layers (standard + advanced)
  * @param convInput  if first layer is Conv2D, set (height, width, channels)
  */
case class SequentialDef(
  inputSize: Int,
  layers: List[AnyLayer],
  optimizer: OptimizerDef = OptimizerDef.Adam(),
  seed: Long = 42L,
  convInput: Option[ConvInput] = None
)

object SequentialDef:
  def apply(inputSize: Int, layers: List[LayerDef]): SequentialDef =
    SequentialDef(inputSize, layers.map(AnyLayer.Standard(_)))
  def apply(inputSize: Int, layers: List[LayerDef], optimizer: OptimizerDef): SequentialDef =
    SequentialDef(inputSize, layers.map(AnyLayer.Standard(_)), optimizer)
  def apply(inputSize: Int, layers: List[LayerDef], optimizer: OptimizerDef, seed: Long): SequentialDef =
    SequentialDef(inputSize, layers.map(AnyLayer.Standard(_)), optimizer, seed)

/** Spatial input shape for convolutional models. */
case class ConvInput(height: Int, width: Int, channels: Int)

/** Functional model: named inputs → DAG of layers → named outputs.
  * Use when you need multi-input, skip connections, or branching.
  * Maps to DL4J ComputationGraph.
  */
case class FunctionalDef(
  inputs: List[String],
  layers: Map[String, LayerDef],
  connections: List[(String, String)],
  outputs: List[String],
  optimizer: OptimizerDef = OptimizerDef.Adam(),
  seed: Long = 42L
)

/** Top-level model definition. */
enum ModelDef:
  case Sequential(arch: SequentialDef)
  case Functional(arch: FunctionalDef)

/** Result of a fit() call — unified across backends.
  *
  * @param loss            final training loss
  * @param epochs          total epochs trained
  * @param lossHistory     per-epoch loss values (empty for simple fit calls)
  * @param validationLoss  loss on validation data (set when validation split used)
  */
case class FitResult(
  loss: Double,
  epochs: Int,
  lossHistory: List[Double] = Nil,
  validationLoss: Option[Double] = None
)

/** Training parameters — loaded from config alongside architecture.
  * Use with ConfigLoader.fromHoconWithTraining().
  */
case class TrainingParams(
  epochs: Int = 100,
  learningRate: Double = 0.01,
  batchSize: Int = 32
)

// ═══════════════════════════════════════════════════════════
//  Training Callbacks (ZIO-native)
// ═══════════════════════════════════════════════════════════

/** Events emitted during a training loop. */
enum TrainingEvent:
  /** Fired after each epoch completes — epoch is 1-based. */
  case EpochEnd(epoch: Int, loss: Double, epochTimeMs: Long)
  /** Fired after validation run (only when validation data provided). */
  case ValidationEnd(epoch: Int, validationLoss: Double)
  /** Fired when training finishes. */
  case TrainEnd(result: FitResult)

/** ZIO-native callback for training lifecycle hooks.
  *
  * Implementations receive [[TrainingEvent]]s during `fitWithCallbacksZ`.
  * Return `ZIO.unit` for events you don't handle.
  *
  * {{{
  *   val logger = new TrainingCallback {
  *     def onEvent(event: TrainingEvent) = event match
  *       case TrainingEvent.EpochEnd(ep, loss, ms) =>
  *         ZIO.logInfo(s"Epoch $ep: loss=$loss (${ms}ms)")
  *       case _ => ZIO.unit
  *   }
  * }}}
  */
trait TrainingCallback:
  def onEvent(event: TrainingEvent): UIO[Unit]

/** Early stopping callback — stops training when validation loss plateaus.
  *
  * Tracks validation loss across epochs. If it fails to improve by at least
  * `minDelta` for `patience` consecutive epochs, sets `shouldStop = true`.
  * The training loop checks this after each epoch's validation run.
  *
  * {{{
  *   val earlyStop = EarlyStopping(patience = 5, minDelta = 0.001)
  *   model.fitWithCallbacksZ(feats, labels, 100, callbacks = List(earlyStop))
  * }}}
  */
class EarlyStopping(val patience: Int = 5, val minDelta: Double = 0.001) extends TrainingCallback:
  private var bestLoss = Double.MaxValue
  private var stalled = 0
  @volatile private var _shouldStop = false
  def shouldStop: Boolean = _shouldStop

  override def onEvent(event: TrainingEvent): UIO[Unit] = event match
    case TrainingEvent.ValidationEnd(_, vl) => ZIO.succeed {
      if vl < bestLoss - minDelta then
        bestLoss = vl; stalled = 0
      else
        stalled += 1
        if stalled >= patience then _shouldStop = true
    }
    case _ => ZIO.unit

  /** Reset internal state — call between independent training runs. */
  def reset(): Unit =
    bestLoss = Double.MaxValue; stalled = 0; _shouldStop = false

/** Learning rate schedule: a function from (epochDone, currentLR) to newLR.
  * Used by `fitWithCallbacksZ` when a schedule is provided.
  *
  * {{{
  *   // Cosine annealing: cycles between 0.0001 and 0.01 every 10 epochs
  *   val cosine = LRSchedule.cosineAnnealing(minLr = 0.0001f, maxLr = 0.01f, cycleLength = 10)
  *   model.fitWithCallbacksZ(feats, labels, 100, lr = 0.01f, lrSchedule = cosine)
  * }}}
  */
object LRSchedule:
  /** Fixed learning rate — no change between epochs (default). */
  val fixed: (Int, Float) => Float = (_, lr) => lr

  /** Cosine annealing schedule.
    * @param minLr       minimum learning rate (trough)
    * @param maxLr       maximum learning rate (peak)
    * @param cycleLength epochs per half-cycle (peak-to-trough)
    */
  def cosine(minLr: Float, maxLr: Float, cycleLength: Int): (Int, Float) => Float =
    (epoch, _) =>
      val progress = (epoch % cycleLength).toDouble / cycleLength
      (minLr + (maxLr - minLr) * ((1.0 + math.cos(math.Pi * progress)) / 2.0)).toFloat
