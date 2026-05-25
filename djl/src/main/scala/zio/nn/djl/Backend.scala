package zio.nn.djl

import zio.nn.*
import ai.djl.nn.{Block, SequentialBlock, Blocks, Activation => DJLActivation, LambdaBlock}
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.LSTM as DJLLSTM
import ai.djl.nn.recurrent.GRU as DJLGRU
import ai.djl.nn.norm.BatchNorm as DJLBN
import ai.djl.nn.transformer.{IdEmbedding, ScaledDotProductAttentionBlock}
import ai.djl.ndarray.{NDList, NDManager}
import ai.djl.training.ParameterStore
import scala.collection.mutable

/** Compiles a [[ModelDef]] into a DJL [[Block]].
  * Usage: `Backend.compile(myArch)` → ready for `ZModel.create(block, name)`.
  * Escape hatch: the returned `Block` is raw DJL — cast, inspect, or pass to any DJL API.
  */
object Backend:

  def compile(model: ModelDef): Block = model match
    case ModelDef.Sequential(arch)  => compileSequential(arch)
    case ModelDef.Functional(arch)  => compileFunctional(arch)

  private def compileSequential(arch: SequentialDef): Block =
    val block = new SequentialBlock()
    arch.layers.foreach {
      case AnyLayer.Standard(layer) => block.add(toDJLBlock(layer))
      case AnyLayer.Advanced(adv)  => block.add(toDJLAdvancedLayer(adv))
    }
    block

  /** Compiles a FunctionalDef DAG into a DJL LambdaBlock.
    * Topological sort + forward-pass routing via DJL's LambdaBlock.
    *
    * Handles: multi-input merge (concatenation), fan-out, skip connections.
    * Not supported: recurrent edges / cycles.
    */
  private def compileFunctional(arch: FunctionalDef): Block =
    val layerBlocks = arch.layers.map((name, layer) => name -> toDJLBlock(layer))

    // Build adjacency for topological sort (Kahn's algorithm)
    val inDegree = mutable.Map[String, Int]()
    val consumers = mutable.Map[String, List[String]]().withDefaultValue(Nil)
    arch.connections.foreach { (src, tgt) =>
      inDegree(tgt) = inDegree.getOrElse(tgt, 0) + 1
      consumers(src) = tgt :: consumers(src)
    }
    arch.inputs.foreach(in => inDegree.getOrElseUpdate(in, 0))
    arch.outputs.foreach(out => inDegree.getOrElseUpdate(out, 0))

    // Validate: all layer nodes must appear in connections
    val allNodes = (arch.inputs ++ arch.layers.keys ++ arch.outputs).toSet
    arch.connections.foreach { (src, tgt) =>
      require(allNodes.contains(src), s"FunctionalDef: unknown source '$src'")
      require(allNodes.contains(tgt), s"FunctionalDef: unknown target '$tgt'")
    }

    // Kahn's topological sort
    val queue = mutable.Queue.from(arch.inputs.filter(inDegree(_) == 0))
    val sorted = mutable.ListBuffer[String]()
    while queue.nonEmpty do
      val node = queue.dequeue()
      sorted += node
      for tgt <- consumers(node) do
        inDegree(tgt) -= 1
        if inDegree(tgt) == 0 then queue.enqueue(tgt)

    val layerOrder = sorted.filter(layerBlocks.contains).toList
    require(layerOrder.size == layerBlocks.size,
      s"FunctionalDef: cycle detected or unreachable layers. Sorted ${layerOrder.size}/${layerBlocks.size}")

    // Build producer map: target → list of sources
    val producers = mutable.Map[String, List[String]]().withDefaultValue(Nil)
    arch.connections.foreach((src, tgt) => producers(tgt) = src :: producers(tgt))

    // LambdaBlock that executes the DAG
    val ndmRef = new java.util.concurrent.atomic.AtomicReference[NDManager]()

    new LambdaBlock(
      new java.util.function.Function[NDList, NDList] {
        def apply(inputs: NDList): NDList =
          val ndm = if ndmRef.get() == null then
            val m = NDManager.newBaseManager()
            ndmRef.set(m); m
          else ndmRef.get()

          val values = mutable.Map[String, NDList]()
          // Initialize input nodes
          arch.inputs.zipWithIndex.foreach((name, i) => values(name) = new NDList(inputs.get(i)))

          val ps = new ParameterStore(ndm, false)

          // Execute layers in topological order
          for node <- layerOrder do
            val block = layerBlocks(node)
            val srcs = producers(node)
            val blockInput = srcs match
              case Nil =>
                new NDList(values(arch.inputs.head).singletonOrThrow())
              case src :: Nil =>
                values.getOrElse(src, new NDList())
              case multi =>
                val arrays = multi.flatMap(s => values.get(s).map(_.singletonOrThrow()))
                if arrays.isEmpty then new NDList()
                else if arrays.length == 1 then new NDList(arrays.head)
                else new NDList(ai.djl.ndarray.NDArrays.concat(new NDList(arrays: _*)))

            val output = block.forward(ps, blockInput, false)
            values(node) = output

          // Collect outputs
          val resultArrays = arch.outputs.flatMap(o => values.get(o).map(_.singletonOrThrow()))
          new NDList(resultArrays*)
      }
    )

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

    case LayerDef.Embedding(vocabSize, embeddingDim, _) =>
      new IdEmbedding.Builder()
        .setDictionarySize(vocabSize)
        .setEmbeddingSize(embeddingDim)
        .build()

  private def toDJLActivationBlock(act: ActivationFn): Block = act match
    case ActivationFn.Tanh      => DJLActivation.tanhBlock()
    case ActivationFn.ReLU      => DJLActivation.reluBlock()
    case ActivationFn.Sigmoid   => DJLActivation.sigmoidBlock()
    case ActivationFn.Softmax   => DJLActivation.reluBlock()
    case ActivationFn.Identity  => Blocks.identityBlock()
    case ActivationFn.LeakyReLU => DJLActivation.leakyReluBlock(0.01f)

  private def toDJLAdvancedLayer(adv: AdvancedLayerDef): Block = adv match
    case AdvancedLayerDef.GRU(_, nOut, _, dropout) =>
      DJLGRU.builder().setNumLayers(1).setStateSize(nOut)
        .optDropRate(dropout.toFloat).build()
    case AdvancedLayerDef.BiDirectional(kind, _, nOut, _, dropout) =>
      kind match
        case BidirectionalKind.LSTM =>
          DJLLSTM.builder().setNumLayers(1).setStateSize(nOut)
            .optDropRate(dropout.toFloat).optBidirectional(true).build()
        case BidirectionalKind.GRU =>
          DJLGRU.builder().setNumLayers(1).setStateSize(nOut)
            .optDropRate(dropout.toFloat).optBidirectional(true).build()
    case AdvancedLayerDef.MultiHeadAttention(embeddingDim, numHeads, dropout) =>
      ScaledDotProductAttentionBlock.builder()
        .setEmbeddingSize(embeddingDim)
        .setHeadCount(numHeads)
        .optAttentionProbsDropoutProb(dropout.toFloat)
        .build()
