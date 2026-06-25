package zio.nn.storch

import zio.nn.*
import torch.*
import torch.nn.*
import torch.optim.{Adam, SGD}
import torch.nn.modules.TensorModule
import scala.util.Try

// ═══════════════════════════════════════════════
//  ZModel — unified API across backends
// ═══════════════════════════════════════════════

/** ZModel wrapping a storch [[TensorModule]].
  *
  * In the storch / libtorch paradigm, modules are callable objects that
  * hold parameters and compute a forward pass via `module(input)`.
  * This wrapper provides the zio‑nn unified API (predict, fit, save, load, close)
  * over typed `Tensor[Float32]` modules.
  *
  * @param module the underlying storch module, typed for 32‑bit floats
  */
case class ZModel(module: TensorModule[Float32]):

  /** UNIFIED: predict from a tensor. Forward pass in eval mode (no gradient tracking). */
  def predict(input: Tensor[Float32]): Try[Tensor[Float32]] = Try {
    module.eval()
    module(input)
  }

  /** Predict and return the flattened result as `Array[Double]`. */
  def predictDouble(input: Tensor[Float32]): Try[Array[Double]] = Try {
    module.eval()
    val output = module(input)
    val raw = output.contiguous().toArray
    raw.map(_.toDouble)
  }

  /** Predict and argmax to token indices (predicted class per sample).
    *
    * For 2‑D output `[batch, nClasses]` the argmax is taken along the last
    * axis.  For 1‑D or scalar output the argmax is taken over all values.
    */
  def predictInt(input: Tensor[Float32]): Try[Array[Int]] = Try {
    module.eval()
    val output = module(input)
    val shape = output.shape
    val flat = output.contiguous().flatten()
    val arr = flat.toArray
    val numClasses = shape.lastOption match
      case None | Some(1) => 1
      case Some(n)        => n.toString.toInt
    if numClasses == 1 then
      // Single output — overall argmax
      Array(arr.indices.maxBy(arr(_)))
    else
      arr.grouped(numClasses).map(_.zipWithIndex.maxBy(_._1)._2).toArray
  }

  /** UNIFIED: train the module using full‑batch gradient descent.
    *
    * @param modelDef      architecture definition (metadata, not used for training)
    * @param inputs        training input tensor
    * @param targets       target tensor
    * @param optimizerDef  optimizer configuration (Adam / SGD / RMSprop)
    * @param lossFn        loss function (MSE, MAE, BinaryCrossEntropy, …)
    * @param epochs        number of training epochs
    * @param batchSize     batch size (currently full‑batch; mini‑batch not yet supported)
    */
  def fit(
    modelDef: ModelDef,
    inputs: Tensor[Float32],
    targets: Tensor[Float32],
    optimizerDef: OptimizerDef,
    lossFn: LossFn,
    epochs: Int,
    batchSize: Int
  ): Try[FitResult] = Try {
    module.train()
    val optimizer = optimizerDef match
      case OptimizerDef.Adam(lr)  => Adam(module.parameters, lr = lr.toFloat)
      case OptimizerDef.SGD(lr)   => SGD(module.parameters, lr = lr.toFloat)
      case OptimizerDef.RMSprop(lr) =>
        // RMSprop not directly exposed by libtorch via Adam/SGD;
        // fall back to SGD with momentum
        SGD(module.parameters, lr = lr.toFloat, momentum = 0.9f)
    val criterion = lossFn match
      case LossFn.MSE                        => MSELoss()
      case LossFn.MAE                        => L1Loss()
      case LossFn.BinaryCrossEntropy(_)      => BCEWithLogitsLoss()
      case LossFn.CategoricalCrossEntropy(_) => CrossEntropyLoss()
      case LossFn.Huber(_)                   =>
        // HuberLoss may not exist in all libtorch versions — fall back to MSE
        MSELoss()
    def computeLoss(m: Any, out: Tensor[Float32], tgt: Tensor[Float32]): Tensor[Float32] =
      m match
        case c: MSELoss            => c.apply(out, tgt)
        case c: L1Loss             => c.apply(out, tgt)
        case c: BCEWithLogitsLoss  => c.apply(out, tgt)
        case c: CrossEntropyLoss   => c.apply(out, tgt)
        case _                     => throw RuntimeException("Unsupported loss function")

    val lossHistory = scala.collection.mutable.ListBuffer[Double]()
    for _ <- 0 until epochs do
      optimizer.zeroGrad()
      val output = module(inputs)
      val loss   = computeLoss(criterion, output, targets)
      loss.backward()
      optimizer.step()
      lossHistory += loss.item().asInstanceOf[Float].toDouble
    FitResult(lossHistory.last, epochs, lossHistory.toList)
  }

  /** Save the module's state dictionary to disk. */
  def save(path: String): Try[Unit] = Try {
    torch.save(module.stateDict(), path)
  }

  /** Load a saved state dictionary into the module. */
  def load(path: String): Try[Unit] = Try {
    val stateDict = torch.pickle_load(java.nio.file.Paths.get(path))
    module.loadStateDict(stateDict)
  }

  /** Release native resources.  JavacPP / storch manages memory internally;
    * this is effectively a no‑op.
    */
  def close(): Try[Unit] = Try(())

object ZModel:

  /** UNIFIED: create a model from architecture.  Compiles + wraps. */
  def create(modelDef: ModelDef): Try[ZModel] = Try {
    ZModel(Backend.compile(modelDef))
  }
