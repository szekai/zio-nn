package zio.nn.dl4j

import zio.nn.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{
  DenseLayer, OutputLayer, LSTM => DL4JLSTM,
  BatchNormalization => DL4JBN, DropoutLayer,
  EmbeddingSequenceLayer, SelfAttentionLayer
}
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation as DL4JActivation
import org.nd4j.linalg.learning.config.{Adam => DL4JAdam, Nesterovs => DL4JSGD, RmsProp => DL4JRMSprop}
import org.nd4j.linalg.lossfunctions.LossFunctions

/** Compiles a [[ModelDef]] into a DL4J [[MultiLayerNetwork]].
  * Usage: `Backend.compile(myArch)` → `ZModel.wrap(net)`.
  */
object Backend:

  def compile(model: ModelDef): MultiLayerNetwork = model match
    case ModelDef.Sequential(arch) => compileSequential(arch)
    case ModelDef.Functional(_)    => sys.error("For FunctionalDef, use Backend.compileGraph() which returns ComputationGraph")

  def compileGraph(arch: FunctionalDef): org.deeplearning4j.nn.graph.ComputationGraph =
    val builder = new NeuralNetConfiguration.Builder()
      .seed(arch.seed).weightInit(WeightInit.XAVIER).updater(toDL4J(arch.optimizer))
      .graphBuilder().addInputs(arch.inputs*)
    var g = builder
    for (source, target) <- arch.connections do
      val layer = arch.layers(target)
      g = g.addLayer(target, toDL4JLayer(layer), source)
    g = g.setOutputs(arch.outputs*)
    val graph = new org.deeplearning4j.nn.graph.ComputationGraph(g.build())
    graph.init()
    graph

  private def compileSequential(arch: SequentialDef): MultiLayerNetwork =
    val builder = new NeuralNetConfiguration.Builder()
      .seed(arch.seed).weightInit(WeightInit.XAVIER).updater(toDL4J(arch.optimizer)).list()
    val configured = arch.layers.zipWithIndex.foldLeft(builder) { case (b, (anyLayer, idx)) =>
      b.layer(idx, toDL4JAnyLayer(anyLayer))
    }
    val withInput = arch.convInput match
      case Some(ConvInput(h, w, c)) => configured.setInputType(InputType.convolutional(h, w, c))
      case None                     => configured
    val model = new MultiLayerNetwork(withInput.build())
    model.init()
    model

  private def toDL4JAnyLayer(anyLayer: AnyLayer) = anyLayer match
    case AnyLayer.Standard(layer) => toDL4JLayer(layer)
    case AnyLayer.Advanced(adv)  => toDL4JAdvancedLayer(adv)

  private def toDL4J(opt: OptimizerDef) = opt match
    case OptimizerDef.Adam(lr)    => new DL4JAdam(lr)
    case OptimizerDef.SGD(lr)     => new DL4JSGD(lr)
    case OptimizerDef.RMSprop(lr) => new DL4JRMSprop(lr)

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

    case LayerDef.Conv2D(nIn, filters, kernel, stride, act) =>
      new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(kernel._1, kernel._2)
        .nIn(nIn).nOut(filters).stride(stride._1, stride._2)
        .activation(toDL4JActivation(act)).build()

    case LayerDef.MaxPool2D(poolSize) =>
      new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(
        org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.MAX,
        Array(poolSize._1, poolSize._2)).build()

    case LayerDef.LayerNorm(_) =>
      sys.error("LayerNorm is not available in DL4J 1.0.0-M2.1. Use the DJL backend for native LayerNorm support.")

    case LayerDef.Embedding(vocabSize, embeddingDim, pretrained) =>
      val builder = new EmbeddingSequenceLayer.Builder().nIn(vocabSize).nOut(embeddingDim)
      pretrained match
        case Some(_) =>
          builder.weightInit(WeightInit.ZERO).build()
        case None =>
          builder.weightInit(WeightInit.XAVIER).build()

  private def toDL4JActivation(act: ActivationFn): DL4JActivation = act match
    case ActivationFn.Tanh => DL4JActivation.TANH; case ActivationFn.ReLU => DL4JActivation.RELU
    case ActivationFn.Sigmoid => DL4JActivation.SIGMOID; case ActivationFn.Softmax => DL4JActivation.SOFTMAX
    case ActivationFn.Identity => DL4JActivation.IDENTITY; case ActivationFn.LeakyReLU(_) => DL4JActivation.LEAKYRELU

  private def toDL4JAdvancedLayer(adv: AdvancedLayerDef) = adv match
    case AdvancedLayerDef.GRU(_, _, _, _) =>
      sys.error("GRU is not available in DL4J 1.0.0-M2.1. Use LSTM or switch to DJL backend.")
    case AdvancedLayerDef.BiDirectional(kind, nIn, nOut, act, drop) =>
      val inner = kind match
        case BidirectionalKind.LSTM => new DL4JLSTM.Builder().nIn(nIn).nOut(nOut).activation(toDL4JActivation(act)).build()
        case BidirectionalKind.GRU  => sys.error("GRU is not available in DL4J 1.0.0-M2.1.")
      new Bidirectional(Bidirectional.Mode.CONCAT, inner)
    case AdvancedLayerDef.MultiHeadAttention(embeddingDim, numHeads, dropout) =>
      new SelfAttentionLayer.Builder()
        .nIn(embeddingDim).nOut(embeddingDim)
        .nHeads(numHeads).headSize(embeddingDim / numHeads)
        .projectInput(true).build()
    case AdvancedLayerDef.TransformerEncoder(_, _, _, _, _) =>
      sys.error("TransformerEncoder is not available in DL4J 1.0.0-M2.1. " +
        "Use the DJL backend for native Transformer support, or manually compose " +
        "MultiHeadAttention + Dense + LayerNorm via FunctionalDef.")

  private def toDL4JLoss(loss: LossFn): LossFunctions.LossFunction = loss match
    case LossFn.MSE => LossFunctions.LossFunction.MSE; case LossFn.MAE => LossFunctions.LossFunction.L1
    case LossFn.BinaryCrossEntropy(_) => LossFunctions.LossFunction.XENT
    case LossFn.CategoricalCrossEntropy(_) => LossFunctions.LossFunction.MCXENT
    case LossFn.Huber(_) => LossFunctions.LossFunction.L2 // DL4J has no native Huber — L2 is closest
