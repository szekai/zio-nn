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
  /** Long Short-Term Memory recurrent layer. */
  case LSTM(nIn: Int, nOut: Int, activation: ActivationFn = ActivationFn.Tanh, dropout: Double = 0.0)

  /** Fully-connected (dense) layer. */
  case Dense(nIn: Int, nOut: Int, activation: ActivationFn = ActivationFn.ReLU)

  /** Output layer (regression or classification). Loss function specified here. */
  case Output(nIn: Int, nOut: Int, loss: LossFn = LossFn.MSE, activation: ActivationFn = ActivationFn.Identity)

  /** Batch normalization. */
  case BatchNorm(nIn: Int)

  /** Dropout regularization. */
  case Dropout(rate: Double)

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

/** Sequential model: layers stacked in order. Covers 80% of use cases. */
case class SequentialDef(
  inputSize: Int,
  layers: List[LayerDef],
  optimizer: OptimizerDef = OptimizerDef.Adam(),
  seed: Long = 42L
)

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
