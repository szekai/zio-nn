package zio.nn.storch

import zio.nn.*
import torch.*
import torch.nn.*
import torch.nn.modules.*
import torch.nn.functional as F
import scala.collection.mutable

/** Compiles a [[ModelDef]] into a storch [[TensorModule[Float32]]].
  * Usage: `Backend.compile(myArch)` → ready for forward pass.
  * Escape hatch: the returned `TensorModule[Float32]` is raw storch — cast, inspect, or pass to any storch API.
  */
object Backend:

  def compile(model: ModelDef): TensorModule[Float32] = model match
    case ModelDef.Sequential(arch) => compileSequential(arch)
    case ModelDef.Functional(arch) => compileFunctional(arch)

  private def compileSequential(arch: SequentialDef): TensorModule[Float32] =
    val modules = arch.layers.flatMap {
      case AnyLayer.Standard(layer) => toStorchModule(layer)
      case AnyLayer.Advanced(adv)  => toStorchAdvanced(adv)
    }
    if modules.isEmpty then nn.Sequential[Float32](nn.Identity[Float32]())
    else if modules.length == 1 then modules.head
    else nn.Sequential[Float32](modules*)

  /** Compiles a FunctionalDef DAG into a custom TensorModule.
    * Topological sort + forward-pass routing.
    *
    * Handles: multi-input merge (concatenation), fan-out, skip connections.
    * Not supported: recurrent edges / cycles.
    */
  private def compileFunctional(arch: FunctionalDef): TensorModule[Float32] =
    val subModules = arch.layers.map { (name, layer) =>
      val mods = toStorchModule(layer)
      name -> (if mods.length == 1 then mods.head else nn.Sequential[Float32](mods*): TensorModule[Float32])
    }
    FunctionalModule(subModules, arch.connections, arch.inputs, arch.outputs)

  private def toStorchModule(layer: LayerDef): List[TensorModule[Float32]] = layer match

    case LayerDef.Dense(nIn, nOut, act) =>
      val linear = nn.Linear[Float32](nIn, nOut)
      if act == ActivationFn.Identity then List(linear)
      else List(linear, toActivationModule(act))

    case LayerDef.LSTM(nIn, nOut, act, dropout) =>
      val lstm = LSTMWrapper(nIn, nOut, dropout = dropout.toFloat)
      if act == ActivationFn.Identity then List(lstm)
      else List(lstm, toActivationModule(act))

    case LayerDef.Output(nIn, nOut, _, act) =>
      val linear = nn.Linear[Float32](nIn, nOut)
      if act == ActivationFn.Identity then List(linear)
      else List(linear, toActivationModule(act))

    case LayerDef.BatchNorm(nIn) =>
      List(nn.BatchNorm1d[Float32](nIn))

    case LayerDef.Dropout(rate) =>
      List(nn.Dropout[Float32](rate.toFloat))

    case LayerDef.Conv2D(nIn, filters, kernel, stride, act) =>
      val conv = nn.Conv2d[Float32](nIn, filters, kernel._1, stride = stride._1)
      if act == ActivationFn.Identity then List(conv)
      else List(conv, toActivationModule(act))

    case LayerDef.MaxPool2D(poolSize) =>
      List(nn.MaxPool2d[Float32](poolSize._1, stride = poolSize._1))

    case LayerDef.Flatten =>
      List(nn.Flatten[Float32]())

    case LayerDef.Embedding(vocabSize, dim, _pretrained) =>
      // TODO: handle pretrained weights when Some(EmbeddingWeights)
      List(nn.Embedding[Float32](vocabSize, dim))

    case LayerDef.LayerNorm(nIn) =>
      List(nn.LayerNorm[Float32](nIn))

    case LayerDef.LastTimestep =>
      List(new TensorModule[Float32]:
        def apply(input: Tensor[Float32]): Tensor[Float32] =
          // input: (batch, seqLen, features) → narrow to last step → squeeze → (batch, features)
          input.narrow(1, input.size(1) - 1, 1).squeeze(1)
      )

  private def toStorchAdvanced(adv: AdvancedLayerDef): List[TensorModule[Float32]] = adv match

    case AdvancedLayerDef.GRU(nIn, nOut, _, dropout) =>
      List(GRUWrapper(nIn, nOut, dropout = dropout.toFloat))

    case AdvancedLayerDef.BiDirectional(kind, nIn, nOut, _, dropout) =>
      kind match
        case BidirectionalKind.LSTM =>
          List(LSTMWrapper(nIn, nOut, dropout = dropout.toFloat, bidirectional = true))
        case BidirectionalKind.GRU =>
          List(GRUWrapper(nIn, nOut, dropout = dropout.toFloat, bidirectional = true))

    case AdvancedLayerDef.MultiHeadAttention(embeddingDim, numHeads, dropout) =>
      List(MHAWrapper(embeddingDim, numHeads, dropout = dropout.toFloat))

    case AdvancedLayerDef.TransformerEncoder(dim, numHeads, ffDim, numLayers, dropout) =>
      List(new TransformerEncoder(dim, numHeads, ffDim, numLayers, dropout.toFloat))

  private def toActivationModule(act: ActivationFn): TensorModule[Float32] = act match
    case ActivationFn.Tanh          => nn.Tanh[Float32]()
    case ActivationFn.ReLU          => nn.ReLU[Float32]()
    case ActivationFn.Sigmoid       => nn.Sigmoid[Float32]()
    case ActivationFn.Softmax       => nn.Softmax[Float32](dim = 1)
    case ActivationFn.Identity      => nn.Identity[Float32]()
    case ActivationFn.LeakyReLU(alpha) => nn.LeakyReLU[Float32](negative_slope = alpha.toFloat)

  // ═══════════════════════════════════════════════════════════
  //  Wrapper modules — layers whose forward returns a tuple
  // ═══════════════════════════════════════════════════════════

  /** Wraps [[nn.LSTM]] — forward returns (output, (h_n, c_n)), extracts output. */
  private class LSTMWrapper(
    val inputSize: Int,
    val hiddenSize: Int,
    val numLayers: Int = 1,
    val dropout: Float = 0.0f,
    val batchFirst: Boolean = true,
    val bidirectional: Boolean = false
  ) extends TensorModule[Float32]:
    val lstm: nn.LSTM[Float32] = nn.LSTM[Float32](
      inputSize, hiddenSize, numLayers,
      dropout = dropout, batch_first = batchFirst, bidirectional = bidirectional
    )

    def apply(input: Tensor[Float32]): Tensor[Float32] =
      val batch = input.shape(0).toInt
      val numDirs = if bidirectional then 2 else 1
      val h0 = torch.zeros[Float32](numLayers * numDirs, batch, hiddenSize)
      val c0 = torch.zeros[Float32](numLayers * numDirs, batch, hiddenSize)
      lstm.forward(input, Some((h0, c0)))._1

  /** Wraps [[nn.GRU]] — forward returns (output, h_n), extracts output. */
  private class GRUWrapper(
    val inputSize: Int,
    val hiddenSize: Int,
    val numLayers: Int = 1,
    val dropout: Float = 0.0f,
    val batchFirst: Boolean = true,
    val bidirectional: Boolean = false
  ) extends TensorModule[Float32]:
    val gru: nn.GRU[Float32] = nn.GRU[Float32](
      inputSize, hiddenSize, numLayers,
      dropout = dropout, batch_first = batchFirst, bidirectional = bidirectional
    )

    def apply(input: Tensor[Float32]): Tensor[Float32] =
      gru.forward(input, None)._1

  /** Self-attention using Linear projections + scaled_dot_product_attention.
    * Bypasses storch's broken [[nn.MultiheadAttention]] constructor (checkcast Nothing$ bug).
    */
  private class MHAWrapper(
    val embedDim: Int,
    val numHeads: Int,
    val dropout: Float = 0.0f
  ) extends TensorModule[Float32]:
    private val headDim = embedDim / numHeads
    require(headDim * numHeads == embedDim, s"embedDim=$embedDim must be divisible by numHeads=$numHeads")

    val wq: nn.Linear[Float32] = nn.Linear[Float32](embedDim, embedDim)
    val wk: nn.Linear[Float32] = nn.Linear[Float32](embedDim, embedDim)
    val wv: nn.Linear[Float32] = nn.Linear[Float32](embedDim, embedDim)
    val wo: nn.Linear[Float32] = nn.Linear[Float32](embedDim, embedDim)

    def apply(input: Tensor[Float32]): Tensor[Float32] =
      val (batch, seq) = (input.shape(0).toInt, input.shape(1).toInt)
      val q = wq(input)
      val k = wk(input)
      val v = wv(input)

      def splitHeads(t: Tensor[Float32]): Tensor[Float32] =
        t.reshape(batch, seq, numHeads, headDim).transpose(1, 2)

      val attnOut = F.scaled_dot_product_attention(
        splitHeads(q), splitHeads(k), splitHeads(v),
        attn_mask = None, dropout_p = dropout.toDouble
      )

      val merged = attnOut.transpose(1, 2).reshape(batch, seq, embedDim)
      wo(merged)

  // ═══════════════════════════════════════════════════════════
  //  Transformer components
  // ═══════════════════════════════════════════════════════════

  /** Single Transformer encoder layer: self-attention + feed-forward with residual and layer norm. */
  private class TransformerEncoderLayer(
    dim: Int, numHeads: Int, ffDim: Int, dropout: Float
  ) extends TensorModule[Float32]:
    val selfAttn = MHAWrapper(dim, numHeads, dropout)
    val linear1  = nn.Linear[Float32](dim, ffDim)
    val linear2  = nn.Linear[Float32](ffDim, dim)
    val norm1    = nn.LayerNorm[Float32](dim)
    val norm2    = nn.LayerNorm[Float32](dim)
    val drop     = nn.Dropout[Float32](dropout)

    def apply(input: Tensor[Float32]): Tensor[Float32] =
      // Self-attention sublayer with residual connection + layer norm
      val attnOut = drop(selfAttn(input))
      val x = norm1(input + attnOut)
      // Feed-forward sublayer (Linear → ReLU → Linear → Dropout) with residual + layer norm
      val ffOut = drop(linear2(F.relu(linear1(x))))
      norm2(x + ffOut)

  /** Stacked Transformer encoder layers. */
  private class TransformerEncoder(
    dim: Int, numHeads: Int, ffDim: Int, numLayers: Int, dropout: Float
  ) extends TensorModule[Float32]:
    private val encoderLayers = nn.Sequential[Float32](
      (1 to numLayers).map(_ =>
        new TransformerEncoderLayer(dim, numHeads, ffDim, dropout): TensorModule[Float32]
      )*
    )

    def apply(input: Tensor[Float32]): Tensor[Float32] =
      encoderLayers(input)

  // ═══════════════════════════════════════════════════════════
  //  Functional DAG module
  // ═══════════════════════════════════════════════════════════

  /** Evaluates a DAG of modules via topological sort.
    * Maps to storch's [[TensorModule]] for use in any storch API.
    *
    * @param modules     named sub-modules (layer name → storch module)
    * @param connections directed edges (source → target)
    * @param inputs      input node names (order determines positional input binding)
    * @param outputs     output node names (order determines concatenation order)
    */
  private class FunctionalModule(
    modules: Map[String, TensorModule[Float32]],
    connections: List[(String, String)],
    inputs: List[String],
    outputs: List[String]
  ) extends TensorModule[Float32]:
    val moduleList: ModuleList[Float32] =
      nn.ModuleList[Float32](modules.values.toSeq*)
    val nameToIndex: Map[String, Int] = modules.keys.zipWithIndex.toMap

    private val (executionOrder, producerMap): (List[String], Map[String, List[String]]) =
      val inDegree   = mutable.Map[String, Int]()
      val consumers  = mutable.Map[String, List[String]]().withDefaultValue(Nil)

      connections.foreach { (src, tgt) =>
        inDegree(tgt) = inDegree.getOrElse(tgt, 0) + 1
        consumers(src) = tgt :: consumers(src)
      }
      inputs.foreach(in    => inDegree.getOrElseUpdate(in, 0))
      outputs.foreach(out  => inDegree.getOrElseUpdate(out, 0))

      val queue  = mutable.Queue.from(inputs.filter(inDegree(_) == 0))
      val sorted = mutable.ListBuffer[String]()
      while queue.nonEmpty do
        val node = queue.dequeue()
        sorted += node
        for tgt <- consumers(node) do
          inDegree(tgt) -= 1
          if inDegree(tgt) == 0 then queue.enqueue(tgt)

      val producers = connections.groupBy(_._2).view.mapValues(_.map(_._1)).toMap.withDefaultValue(Nil)
      (sorted.toList, producers)

    private val layerOrder: List[String] = executionOrder.filter(modules.contains)

    require(
      layerOrder.size == modules.size,
      s"FunctionalDef: cycle detected or unreachable layers. Sorted ${layerOrder.size}/${modules.size}"
    )

    def apply(input: Tensor[Float32]): Tensor[Float32] =
      val values = mutable.Map[String, Tensor[Float32]]()
      if inputs.nonEmpty then
        values(inputs.head) = input

      for node <- layerOrder do
        val mod     = moduleList(nameToIndex(node))
        val srcs    = producerMap(node)
        val modInput = srcs match
          case Nil            =>
            values(inputs.head)
          case List(src)     =>
            values.getOrElse(src, throw RuntimeException(s"FunctionalDef: missing input '$src' for node '$node'"))
          case multi          =>
            val tensors = multi.flatMap(s => values.get(s))
            if tensors.isEmpty then
              throw RuntimeException(s"FunctionalDef: no inputs for node '$node'")
            else if tensors.length == 1 then tensors.head
            else torch.cat(Seq(tensors*), dim = -1)

        values(node) = mod(modInput)

      val resultTensors = outputs.flatMap(o => values.get(o))
      if resultTensors.isEmpty then
        throw RuntimeException("FunctionalDef: no outputs produced")
      else if resultTensors.length == 1 then resultTensors.head
      else torch.cat(Seq(resultTensors*), dim = -1)
