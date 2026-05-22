package zio.nn.dl4j

import zio.nn.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{
  DenseLayer, OutputLayer, LSTM => DL4JLSTM,
  BatchNormalization => DL4JBN, DropoutLayer
}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation as DL4JActivation
import org.nd4j.linalg.learning.config.{Adam => DL4JAdam, Nesterovs => DL4JSGD, RmsProp => DL4JRMSprop}
import org.nd4j.linalg.lossfunctions.LossFunctions

/** Compiles a framework-agnostic [[ModelDef]] into a DL4J [[MultiLayerNetwork]].
  *
  * Usage:
  * {{{
  *   val net: MultiLayerNetwork = BackendDL4J.compile(myArch)
  *   DL4JZio.ZModel.wrap(net)
  * }}}
  *
  * == Escape Hatch ==
  * The returned `MultiLayerNetwork` is a raw DL4J object. Cast, inspect,
  * or pass to any DL4J API directly.
  */
object BackendDL4J:

  def compile(model: ModelDef): MultiLayerNetwork = model match
    case ModelDef.Sequential(arch) => compileSequential(arch)
    case ModelDef.Functional(_)    => sys.error("FunctionalDef not yet supported for DL4J backend. Use raw DL4J ComputationGraph instead.")

  private def compileSequential(arch: SequentialDef): MultiLayerNetwork =
    val builder = new NeuralNetConfiguration.Builder()
      .seed(arch.seed)
      .weightInit(WeightInit.XAVIER)
      .updater(toDL4J(arch.optimizer))
      .list()

    val configured = arch.layers.zipWithIndex.foldLeft(builder) { case (b, (layer, idx)) =>
      b.layer(idx, toDL4JLayer(layer))
    }

    val model = new MultiLayerNetwork(configured.build())
    model.init()
    model

  private def toDL4J(opt: OptimizerDef) = opt match
    case OptimizerDef.Adam(lr)   => new DL4JAdam(lr)
    case OptimizerDef.SGD(lr)    => new DL4JSGD(lr)
    case OptimizerDef.RMSprop(lr)=> new DL4JRMSprop(lr)

  private def toDL4JLayer(layer: LayerDef) = layer match
    case LayerDef.LSTM(nIn, nOut, act, _) =>
      new DL4JLSTM.Builder().nIn(nIn).nOut(nOut).activation(toDL4JActivation(act)).build()
    case LayerDef.Dense(nIn, nOut, act) =>
      new DenseLayer.Builder().nIn(nIn).nOut(nOut).activation(toDL4JActivation(act)).build()
    case LayerDef.Output(nIn, nOut, loss, act) =>
      new OutputLayer.Builder(toDL4JLoss(loss)).nIn(nIn).nOut(nOut).activation(toDL4JActivation(act)).build()
    case LayerDef.BatchNorm(nIn) =>
      new DL4JBN.Builder().nIn(nIn).nOut(nIn).build()
    case LayerDef.Dropout(rate) =>
      new DropoutLayer.Builder(rate).build()

  private def toDL4JActivation(act: ActivationFn): DL4JActivation = act match
    case ActivationFn.Tanh      => DL4JActivation.TANH
    case ActivationFn.ReLU      => DL4JActivation.RELU
    case ActivationFn.Sigmoid   => DL4JActivation.SIGMOID
    case ActivationFn.Softmax   => DL4JActivation.SOFTMAX
    case ActivationFn.Identity  => DL4JActivation.IDENTITY
    case ActivationFn.LeakyReLU => DL4JActivation.LEAKYRELU

  private def toDL4JLoss(loss: LossFn): LossFunctions.LossFunction = loss match
    case LossFn.MSE                 => LossFunctions.LossFunction.MSE
    case LossFn.MAE                 => LossFunctions.LossFunction.L1
    case LossFn.BinaryCrossEntropy  => LossFunctions.LossFunction.XENT
    case LossFn.CategoricalCrossEntropy => LossFunctions.LossFunction.MCXENT
    case LossFn.Huber               => LossFunctions.LossFunction.L1 // Huber not in ND4J; L1 is closest
