package zio.nn.dl4j

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.layers.{
  DenseLayer, OutputLayer, BatchNormalization => DL4JBN,
  DropoutLayer, EmbeddingSequenceLayer, FeedForwardLayer
}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional
import org.nd4j.linalg.api.ndarray.INDArray

import java.io.{ByteArrayOutputStream, DataOutputStream}
import java.nio.file.{Files, Path}
import scala.util.Try

/** Exports a DL4J [[MultiLayerNetwork]] to ONNX protobuf format WITHOUT
  * requiring protobuf-java as a dependency.
  *
  * The ONNX binary is written by hand using the protobuf wire format
  * (`java.io.DataOutputStream`). Layer weights and biases are extracted
  * from the DL4J API and mapped to ONNX operators (Gemm, BatchNormalization,
  * LSTM, Gather, activations, etc.).
  *
  * ==Supported Layers==
  *   – [[DenseLayer]]              → Gemm + activation (Relu / Tanh / Sigmoid / Softmax / LeakyRelu)
  *   – [[OutputLayer]]             → Gemm + activation
  *   – [[DL4JBN BatchNormalization]] → BatchNormalization
  *   – [[DropoutLayer]]            → identity at inference (skipped)
  *   – [[LSTM]]                    → LSTM + Squeeze (with DL4J→ONNX gate reorder)
  *   – [[EmbeddingSequenceLayer]]  → Gather
  *   – [[Bidirectional]]           → LSTM with direction=bidirectional
  *
  * ==Usage==
  * {{{
  *   val model: MultiLayerNetwork = ???
  *   Dl4jToOnnx.toOnnx(model) match
  *     case Success(bytes) => Files.write(path, bytes)
  *     case Failure(e)     => println(s"Export failed: $e")
  * }}}
  */
object Dl4jToOnnx:

  // ═══════════════════════════════════════════════════════════
  //  Public API
  // ═══════════════════════════════════════════════════════════

  /** Export a DL4J [[MultiLayerNetwork]] to ONNX protobuf bytes.
    *
    * @param model a trained DL4J model
    * @return raw ONNX protobuf bytes, or `Failure` on error
    */
  def toOnnx(model: MultiLayerNetwork): Try[Array[Byte]] = Try {
    val layers = model.getLayers
    if layers == null || layers.length == 0 then
      throw IllegalArgumentException("Model has no layers")

    val n = layers.length

    val firstConf   = layers(0).conf().getLayer
    val inputDim    = nIn(firstConf)
    val lastConf    = layers(n - 1).conf().getLayer
    val outputDim   = nOut(lastConf)
    val isEmbedding = firstConf.isInstanceOf[EmbeddingSequenceLayer]

    val nodes       = Vector.newBuilder[Array[Byte]]
    val inits       = Vector.newBuilder[Array[Byte]]

    val inputValue  = makeValueInfo("input", if isEmbedding then 7 else 1,
                        if isEmbedding then Array(-1L, -1L) else Array(-1L, inputDim.toLong))
    val outputValue = makeValueInfo("output", 1, Array(-1L, outputDim.toLong))

    var prev = "input"
    for i <- 0 until n do
      val dl4jLayer = layers(i)
      val lConf     = dl4jLayer.conf().getLayer

      lConf match

        case d: DenseLayer =>
          val (wName, bName, gemmOut) = (s"w_$i", s"b_$i", s"gemm_$i")
          emitWeightBias(dl4jLayer, wName, bName, inits)
          nodes += makeNode("Gemm",
            inputs  = Seq(prev, wName, bName),
            outputs = Seq(gemmOut),
            attrs   = Seq(
              makeAttrFloat("alpha", 1.0f),
              makeAttrFloat("beta", 1.0f),
              makeAttrInt("transB", 1L)
            ))
          prev = activationNode(d.getActivationFn, gemmOut, i, n, nodes)

        case o: OutputLayer =>
          val (wName, bName, gemmOut) = (s"w_$i", s"b_$i", s"gemm_$i")
          emitWeightBias(dl4jLayer, wName, bName, inits)
          nodes += makeNode("Gemm",
            inputs  = Seq(prev, wName, bName),
            outputs = Seq(gemmOut),
            attrs   = Seq(
              makeAttrFloat("alpha", 1.0f),
              makeAttrFloat("beta", 1.0f),
              makeAttrInt("transB", 1L)
            ))
          prev = activationNode(o.getActivationFn, gemmOut, i, n, nodes)

        case _: DropoutLayer =>

        case bn: DL4JBN =>
          val nOut  = bn.getNOut
          val sName = s"bn_s_$i"; val bName = s"bn_b_$i"
          val mName = s"bn_m_$i"; val vName = s"bn_v_$i"
          val outNm = s"bn_$i"

          Seq(("gamma", sName), ("beta", bName), ("mean", mName), ("var", vName))
            .foreach { (k, nm) =>
              val p = dl4jLayer.getParam(k)
              if p != null then inits += makeTensor1D(nm, 1, toFloats(p))
            }
          nodes += makeNode("BatchNormalization",
            inputs  = Seq(prev, sName, bName, mName, vName),
            outputs = Seq(outNm),
            attrs   = Seq(makeAttrFloat("epsilon", 1e-5f)))
          prev = outNm

        case lstmConf: LSTM =>
          val hidden = lstmConf.getNOut.toInt
          val wName  = s"lstm_w_$i"; val rName = s"lstm_r_$i"
          val bName  = s"lstm_b_$i"
          val lstmH  = s"lstm_${i}_h"
          val lstmC  = s"lstm_${i}_c"
          val lstmY  = s"lstm_${i}_y"
          val flat   = s"lstm_${i}_flat"

          emitLSTMWeights(dl4jLayer, i, wName, rName, bName, hidden, inits)

          nodes += makeNode("LSTM",
            inputs  = Seq(prev, wName, rName, bName),
            outputs = Seq(lstmY, lstmH, lstmC),
            attrs   = Seq(
              makeAttrInt("hidden_size", hidden.toLong),
              makeAttrString("direction", "forward")
            ))

          val axName = s"lstm_sq_$i"
          inits += makeTensor1DInt64(axName, Array(0L))
          nodes += makeNode("Squeeze", Seq(lstmH, axName), Seq(flat))
          prev = flat

        case bidir: Bidirectional =>
          val hidden = nOut(lConf).toInt
          val wName  = s"bd_w_$i"; val rName = s"bd_r_$i"
          val bName  = s"bd_b_$i"
          val lstmH  = s"bd_${i}_h"
          val lstmC  = s"bd_${i}_c"
          val lstmY  = s"bd_${i}_y"
          val flat   = s"bd_${i}_flat"

          emitLSTMWeights(dl4jLayer, i, wName, rName, bName, hidden, inits)

          nodes += makeNode("LSTM",
            inputs  = Seq(prev, wName, rName, bName),
            outputs = Seq(lstmY, lstmH, lstmC),
            attrs   = Seq(
              makeAttrInt("hidden_size", hidden.toLong),
              makeAttrString("direction", "bidirectional")
            ))

          prev = lstmH

        case emb: EmbeddingSequenceLayer =>
          val wName = s"emb_w_$i"; val outNm = s"emb_$i"
          val w = dl4jLayer.getParam("W")
          if w != null then
            val s = w.shape()
            inits += makeTensor(wName, 1, Array(s(0).toLong, s(1).toLong), toFloats(w))
          nodes += makeNode("Gather", Seq(wName, prev), Seq(outNm),
            attrs = Seq(makeAttrInt("axis", 0L)))
          prev = outNm

        case rnn: RnnOutputLayer =>
          val (wName, bName, gemmOut) = (s"w_$i", s"b_$i", s"gemm_$i")
          emitWeightBias(dl4jLayer, wName, bName, inits)
          nodes += makeNode("Gemm",
            inputs  = Seq(prev, wName, bName),
            outputs = Seq(gemmOut),
            attrs   = Seq(
              makeAttrFloat("alpha", 1.0f),
              makeAttrFloat("beta", 1.0f),
              makeAttrInt("transB", 1L)
            ))
          prev = activationNode(rnn.getActivationFn, gemmOut, i, n, nodes)

        case _ => ()
    end for

    if prev != "output" then
      nodes += makeNode("Identity", Seq(prev), Seq("output"))

    val graphBytes = writeMessage { out =>
      writeMessageField(out, 1, inputValue)
      writeMessageField(out, 2, outputValue)
      nodes.result().foreach { n  => writeMessageField(out, 3, n) }
      writeStringField(out, 4, "dl4j_exported")
      inits.result().foreach { t  => writeMessageField(out, 5, t) }
    }

    writeMessage { out =>
      writeVarintField(out, 1, 9L)
      writeStringField(out, 2, "zio-nn-dl4j")
      writeStringField(out, 3, "1.0.0")
      writeMessageField(out, 7, graphBytes)
      writeMessageField(out, 8, writeMessage { o =>
        writeStringField(o, 1, "")
        writeVarintField(o, 2, 21L)
      })
    }
  }

  /** Export ONNX model to a file. */
  def saveToFile(model: MultiLayerNetwork, path: Path): Try[Unit] =
    toOnnx(model).flatMap { bytes =>
      Try(Files.write(path, bytes))
    }

  // ═══════════════════════════════════════════════════════════════
  //  Protobuf Wire-Format Primitives
  // ═══════════════════════════════════════════════════════════════

  /** Write an unsigned varint. */
  private def writeVarint(out: DataOutputStream, value: Long): Unit =
    var v = value
    while (v & 0xFFFFFFFFFFFFFF80L) != 0L do
      out.writeByte(((v & 0x7FL) | 0x80L).toInt)
      v >>>= 7
    out.writeByte((v & 0x7FL).toInt)

  /** Create a message by writing to a temporary buffer. */
  private def writeMessage(f: DataOutputStream => Unit): Array[Byte] =
    val baos = ByteArrayOutputStream()
    val out  = DataOutputStream(baos)
    f(out)
    out.close()
    baos.toByteArray

  // ── Individual field writers ──────────────────────────────────

  /** Write a varint field (wire type 0). */
  private def writeVarintField(out: DataOutputStream, field: Int, value: Long): Unit =
    writeVarint(out, (field << 3).toLong)
    writeVarint(out, value)

  /** Write a length-delimited string / message field (wire type 2). */
  private def writeStringField(out: DataOutputStream, field: Int, value: String): Unit =
    val utf8 = value.getBytes("UTF-8")
    writeVarint(out, (field << 3 | 2).toLong)
    writeVarint(out, utf8.length.toLong)
    out.write(utf8)

  /** Write a length-delimited sub-message field (wire type 2). */
  private def writeMessageField(out: DataOutputStream, field: Int, value: Array[Byte]): Unit =
    writeVarint(out, (field << 3 | 2).toLong)
    writeVarint(out, value.length.toLong)
    out.write(value)

  /** Write a 32-bit float field (wire type 5). */
  private def writeFloatField(out: DataOutputStream, field: Int, value: Float): Unit =
    writeVarint(out, (field << 3 | 5).toLong)
    out.writeInt(java.lang.Float.floatToRawIntBits(value))

  /** Write packed repeated int64 (wire type 2, data is varints). */
  private def writePackedInt64s(out: DataOutputStream, field: Int, values: Array[Long]): Unit =
    val payload = writeMessage { o => values.foreach(writeVarint(o, _)) }
    writeVarint(out, (field << 3 | 2).toLong)
    writeVarint(out, payload.length.toLong)
    out.write(payload)

  /** Write packed repeated float (wire type 2, data is 4-byte IEEE 754). */
  private def writePackedFloats(out: DataOutputStream, field: Int, values: Array[Float]): Unit =
    val payload = writeMessage { o => values.foreach(v => o.writeInt(java.lang.Float.floatToRawIntBits(v))) }
    writeVarint(out, (field << 3 | 2).toLong)
    writeVarint(out, payload.length.toLong)
    out.write(payload)

  // ═══════════════════════════════════════════════════════════════
  //  ONNX Message Builders
  // ═══════════════════════════════════════════════════════════════

  // ── TensorProto ───────────────────────────────────────────────

  /** Float tensor (used for weights, biases, batchnorm params). */
  private def makeTensor(name: String, dataType: Int, dims: Array[Long], data: Array[Float]): Array[Byte] =
    writeMessage { out =>
      if dims.length > 0 then writePackedInt64s(out, 1, dims)   // dims
      writePackedFloats(out, 4, data)                             // float_data
      writeVarintField(out, 5, dataType.toLong)                   // data_type
      writeStringField(out, 3, name)                              // name
    }

  /** 1-D float tensor shortcut (e.g. bias, batchnorm params). */
  private def makeTensor1D(name: String, dataType: Int, data: Array[Float]): Array[Byte] =
    makeTensor(name, dataType, Array(data.length.toLong), data)

  /** 1-D int64 tensor shortcut (e.g. axes for Squeeze). */
  private def makeTensor1DInt64(name: String, data: Array[Long]): Array[Byte] =
    writeMessage { out =>
      writePackedInt64s(out, 1, Array(data.length.toLong))      // dims
      writePackedInt64s(out, 11, data)                            // int64_data
      writeVarintField(out, 5, 7L)                                // data_type = INT64
      writeStringField(out, 3, name)                              // name
    }

  // ── ValueInfoProto ────────────────────────────────────────────

  private def makeValueInfo(name: String, elemType: Int, dims: Array[Long]): Array[Byte] =
    writeMessage { out =>
      writeStringField(out, 1, name)
      writeMessageField(out, 2, writeTypeProto(elemType, dims))
    }

  private def writeTypeProto(elemType: Int, dims: Array[Long]): Array[Byte] =
    writeMessage { out =>
      writeMessageField(out, 1, writeMessage { o =>
        writeVarintField(o, 1, elemType.toLong)                    // elem_type (field 1 of Tensor)
        writeMessageField(o, 2, writeTensorShapeProto(dims))        // shape (field 2 of Tensor)
      })
    }

  private def writeTensorShapeProto(dims: Array[Long]): Array[Byte] =
    writeMessage { out =>
      dims.foreach { dim =>
        val dimMsg = writeMessage { o =>
          if dim < 0 then
            writeStringField(o, 2, "N")                            // dim_param for dynamic axes
          else
            writeVarintField(o, 1, dim)                            // dim_value
        }
        writeMessageField(out, 1, dimMsg)
      }
    }

  // ── NodeProto ─────────────────────────────────────────────────

  private def makeNode(
    opType: String,
    inputs: Seq[String],
    outputs: Seq[String],
    attrs: Seq[Array[Byte]] = Seq.empty,
    domain: String = ""
  ): Array[Byte] = writeMessage { out =>
    inputs .foreach(in  => writeStringField(out, 1, in))
    outputs.foreach(oup => writeStringField(out, 2, oup))
    writeStringField(out, 4, opType)
    attrs.foreach(a => writeMessageField(out, 5, a))
    if domain.nonEmpty then writeStringField(out, 7, domain)
  }

  // ── AttributeProto builders ───────────────────────────────────

  private def makeAttrFloat(name: String, value: Float): Array[Byte] = writeMessage { out =>
    writeStringField(out, 1, name)
    writeVarintField(out, 20, 1L)      // type = FLOAT
    writeFloatField(out, 2, value)     // f
  }

  private def makeAttrInt(name: String, value: Long): Array[Byte] = writeMessage { out =>
    writeStringField(out, 1, name)
    writeVarintField(out, 20, 2L)      // type = INT
    writeVarintField(out, 3, value)    // i
  }

  private def makeAttrString(name: String, value: String): Array[Byte] = writeMessage { out =>
    writeStringField(out, 1, name)
    writeVarintField(out, 20, 3L)      // type = STRING
    writeStringField(out, 4, value)    // s
  }

  // ═══════════════════════════════════════════════════════════════
  //  DL4J Traversal Helpers
  // ═══════════════════════════════════════════════════════════════

  /** Convert INDArray data to a flat Float array. */
  private def toFloats(arr: INDArray): Array[Float] = arr.data().asFloat()

  /** Extract nIn from a layer configuration. */
  private def nIn(conf: org.deeplearning4j.nn.conf.layers.Layer): Int = conf match
    case f: FeedForwardLayer => f.getNIn.toInt
    case _ => 0

  /** Extract nOut from a layer configuration. */
  private def nOut(conf: org.deeplearning4j.nn.conf.layers.Layer): Int = conf match
    case f: FeedForwardLayer => f.getNOut.toInt
    case _ => 0

  /** Add an activation node after a Gemm/batch-norm and return the output name. */
  private def activationNode(
    act: org.nd4j.linalg.activations.IActivation,
    inputName: String,
    idx: Int,
    totalLayers: Int,
    nodes: scala.collection.mutable.Builder[Array[Byte], Vector[Array[Byte]]]
  ): String =
    val onnxAct = act.getClass.getSimpleName match
      case s if s.contains("ReLU")    => "Relu"
      case s if s.contains("TanH") || s.contains("Tanh") => "Tanh"
      case s if s.contains("Sigmoid") => "Sigmoid"
      case s if s.contains("Softmax") => "Softmax"
      case s if s.contains("LReLU")   => "LeakyRelu"
      case _                          => null

    if onnxAct == null then
      inputName
    else
      val outName = if idx == totalLayers - 1 then "output"
                    else s"${onnxAct.toLowerCase}_${idx}"
      nodes += makeNode(onnxAct, Seq(inputName), Seq(outName))
      outName

  /** Write weight (W) and bias (b) tensors from a DL4J layer. */
  private def emitWeightBias(
    dl: org.deeplearning4j.nn.api.Layer,
    wName: String,
    bName: String,
    inits: scala.collection.mutable.Builder[Array[Byte], Vector[Array[Byte]]]
  ): Unit =
    val w = dl.getParam("W")
    val b = dl.getParam("b")
    if w != null then
      val s = w.shape()
      inits += makeTensor(wName, 1, Array(s(0).toLong, s(1).toLong), toFloats(w))
    if b != null then
      val s = b.shape()
      val dims = if s.length == 2 && s(0) == 1 then Array(s(1).toLong) else s.map(_.toLong)
      inits += makeTensor(bName, 1, dims, toFloats(b))

  /** Write LSTM weight tensors (W, RW, B) with DL4J→ONNX gate reorder. */
  private def emitLSTMWeights(
    dl: org.deeplearning4j.nn.api.Layer,
    idx: Int,
    wName: String,
    rName: String,
    bName: String,
    hidden: Int,
    inits: scala.collection.mutable.Builder[Array[Byte], Vector[Array[Byte]]]
  ): Unit =
    // DL4J LSTM stores gates in IFOC order: Input, Forget, Output, Cell
    // ONNX expects IOFC order: Input, Output, Forget, Cell
    val w  = dl.getParam("W")
    val rw = dl.getParam("RW")
    val b  = dl.getParam("b")

    if w != null then
      val data = toFloats(w)
      val reord = reorderGates(data, hidden, nIn(dl.conf().getLayer).toInt)
      inits += makeTensor(wName, 1, Array(1L, hidden * 4L, nIn(dl.conf().getLayer).toLong), reord)

    if rw != null then
      val data = toFloats(rw)
      val reord = reorderGates(data, hidden, hidden)
      inits += makeTensor(rName, 1, Array(1L, hidden * 4L, hidden.toLong), reord)

    if b != null then
      val data = toFloats(b)
      val reord = reorderGates(data, hidden, 1)
      val onnxBias = new Array[Float](hidden * 8)
      System.arraycopy(reord, 0, onnxBias, 0, hidden * 4)
      System.arraycopy(reord, 0, onnxBias, hidden * 4, hidden * 4)
      inits += makeTensor(bName, 1, Array(1L, hidden * 8L), onnxBias)

  /** Reorder LSTM gates from DL4J (IFOC) to ONNX (IOFC).
    *
    * DL4J order: Input[0:n], Forget[n:2n], Output[2n:3n], Cell[3n:4n]
    * ONNX  order: Input[0:n], Output[n:2n], Forget[2n:3n], Cell[3n:4n]
    *
    * Swaps the Forget and Output gate partitions.
    *
    * @param data   flat float array of weights or biases
    * @param nOut   hidden size (number of units per gate)
    * @param dim    columns per row (input size for W, nOut for RW, 1 for bias)
    */
  private def reorderGates(data: Array[Float], nOut: Int, dim: Int): Array[Float] =
    val result  = java.util.Arrays.copyOf(data, data.length)
    val gs      = nOut * dim   // gate partition size (in floats)

    // Swap gate index 1 (Forget) ↔ gate index 2 (Output)
    val tmp = new Array[Float](gs)
    System.arraycopy(result, gs,      tmp, 0, gs)           // save Forget
    System.arraycopy(result, 2 * gs,  result, gs, gs)       // Output → position 1
    System.arraycopy(tmp,     0,      result, 2 * gs, gs)   // Forget → position 2
    result

end Dl4jToOnnx
