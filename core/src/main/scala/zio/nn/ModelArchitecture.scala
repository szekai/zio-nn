package zio.nn

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

/** Sequential model: layers stacked in order. Covers 80% of use cases.
  *
  * @param inputSize  input dimension (features for flat, channels for Conv2D)
  * @param convInput  if first layer is Conv2D, set (height, width, channels)
  *                   so the backend can configure InputType.convolutional
  */
case class SequentialDef(
  inputSize: Int,
  layers: List[LayerDef],
  optimizer: OptimizerDef = OptimizerDef.Adam(),
  seed: Long = 42L,
  convInput: Option[ConvInput] = None
)

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

/** Result of a fit() call — unified across backends. */
case class FitResult(loss: Double, epochs: Int)

/** Training parameters — loaded from config alongside architecture.
  * Use with ConfigLoader.fromHoconWithTraining().
  */
case class TrainingParams(
  epochs: Int = 100,
  learningRate: Double = 0.01,
  batchSize: Int = 32
)
