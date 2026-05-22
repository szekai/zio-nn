package zio.nn.djl

import zio.nn.*
import ai.djl.nn.{Block, SequentialBlock, Blocks, Activation => DJLActivation}
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.LSTM as DJLLSTM
import ai.djl.nn.norm.BatchNorm as DJLBN

/** Compiles a [[ModelDef]] into a DJL [[Block]].
  * Usage: `Backend.compile(myArch)` → ready for `ZModel.create(block, name)`.
  * Escape hatch: the returned `Block` is raw DJL — cast, inspect, or pass to any DJL API.
  */
object Backend:

  def compile(model: ModelDef): Block = model match
    case ModelDef.Sequential(arch) => compileSequential(arch)
    case ModelDef.Functional(_)    => sys.error("FunctionalDef not yet supported for DJL. Use raw DJL Block instead.")

  private def compileSequential(arch: SequentialDef): Block =
    val block = new SequentialBlock()
    arch.layers.foreach(layer => block.add(toDJLBlock(layer)))
    block

  private def toDJLBlock(layer: LayerDef): Block = layer match
    case LayerDef.LSTM(nIn, nOut, act, dropout) =>
      DJLLSTM.builder().setNumLayers(1).setStateSize(nOut).optDropRate(dropout.toFloat).build()

    case LayerDef.Dense(nIn, nOut, act) =>
      val dense = Linear.builder().setUnits(nOut.toLong).build()
      if act == ActivationFn.Identity then dense
      else new SequentialBlock().add(dense).add(toDJLActivationBlock(act))

    case LayerDef.Output(nIn, nOut, _loss, act) =>
      val linear = Linear.builder().setUnits(nOut.toLong).build()
      if act == ActivationFn.Identity then linear
      else new SequentialBlock().add(linear).add(toDJLActivationBlock(act))

    case LayerDef.BatchNorm(nIn) => DJLBN.builder().build()

    case LayerDef.Dropout(_rate) => Blocks.identityBlock()

    case LayerDef.Conv2D(_, filters, kernel, stride, act) =>
      val conv = ai.djl.nn.convolutional.Conv2d.builder()
        .setFilters(filters)
        .setKernelShape(new ai.djl.ndarray.types.Shape(kernel._1, kernel._2))
        .optStride(new ai.djl.ndarray.types.Shape(stride._1, stride._2))
        .build()
      if act == ActivationFn.Identity then conv
      else new SequentialBlock().add(conv).add(toDJLActivationBlock(act))

    case LayerDef.MaxPool2D(poolSize) =>
      ai.djl.nn.pooling.Pool.maxPool2dBlock(new ai.djl.ndarray.types.Shape(poolSize._1, poolSize._2))

    case LayerDef.Flatten =>
      Blocks.batchFlattenBlock()

  private def toDJLActivationBlock(act: ActivationFn): Block = act match
    case ActivationFn.Tanh      => DJLActivation.tanhBlock()
    case ActivationFn.ReLU      => DJLActivation.reluBlock()
    case ActivationFn.Sigmoid   => DJLActivation.sigmoidBlock()
    case ActivationFn.Softmax   => DJLActivation.reluBlock()
    case ActivationFn.Identity  => Blocks.identityBlock()
    case ActivationFn.LeakyReLU => DJLActivation.leakyReluBlock(0.01f)
