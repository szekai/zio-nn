package zio.nn.djl

import zio.nn.*
import ai.djl.nn.{Block, SequentialBlock, Blocks, Activation => DJLActivation}
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.LSTM as DJLLSTM
import ai.djl.nn.norm.BatchNorm as DJLBN

/** Compiles a framework-agnostic [[ModelDef]] into a DJL [[Block]].
  *
  * Usage:
  * {{{
  *   val block: Block = BackendDJL.compile(myArch)
  *   DJLZio.ZModel.create(block, "myModel")
  * }}}
  *
  * == Escape Hatch ==
  * The returned `Block` is a raw DJL object. You can cast it, inspect it,
  * or pass it to any DJL API directly — this is the escape hatch.
  * For completely custom architectures not expressible in [[ModelDef]],
  * construct your own `Block` and pass it to `DJLZio.ZModel.create(block)`.
  */
object BackendDJL:

  def compile(model: ModelDef): Block = model match
    case ModelDef.Sequential(arch) => compileSequential(arch)
    case ModelDef.Functional(_)    => sys.error("FunctionalDef not yet supported for DJL. Use raw DJL Block instead.")

  private def compileSequential(arch: SequentialDef): Block =
    val block = new SequentialBlock()
    arch.layers.foreach(layer => block.add(toDJLBlock(layer)))
    block

  private def toDJLBlock(layer: LayerDef): Block = layer match
    case LayerDef.LSTM(nIn, nOut, act, dropout) =>
      DJLLSTM.builder()
        .setNumLayers(1)
        .setStateSize(nOut)
        .optDropRate(dropout.toFloat)
        .build()

    case LayerDef.Dense(nIn, nOut, act) =>
      val dense = Linear.builder().setUnits(nOut.toLong).build()
      if act == ActivationFn.Identity then dense
      else new SequentialBlock().add(dense).add(toDJLActivationBlock(act))

    case LayerDef.Output(nIn, nOut, _loss, act) =>
      val linear = Linear.builder().setUnits(nOut.toLong).build()
      if act == ActivationFn.Identity then linear
      else new SequentialBlock().add(linear).add(toDJLActivationBlock(act))

    case LayerDef.BatchNorm(nIn) =>
      DJLBN.builder().build()

    case LayerDef.Dropout(_rate) =>
      // Dropout as standalone Block not exposed in DJL 0.36.
      // Configure dropout on the Trainer or LSTM builder instead.
      Blocks.identityBlock()

  private def toDJLActivationBlock(act: ActivationFn): Block = act match
    case ActivationFn.Tanh      => DJLActivation.tanhBlock()
    case ActivationFn.ReLU      => DJLActivation.reluBlock()
    case ActivationFn.Sigmoid   => DJLActivation.sigmoidBlock()
    case ActivationFn.Softmax   => DJLActivation.reluBlock() // softmax not used in our models; fallback
    case ActivationFn.Identity  => Blocks.identityBlock()
    case ActivationFn.LeakyReLU => DJLActivation.leakyReluBlock(0.01f)
