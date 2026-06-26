package zio.nn.dl4j

import zio.*
import zio.stream.*
import zio.nn.{EncodingResult, FitResult, TrainingCallback, TrainingEvent, EarlyStopping, LRSchedule}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.nio.file.Path

object zioApi:

  private def checkpointLoop(
    fitChunk: Int => ZIO[Any, Throwable, FitResult],
    saveCheckpoint: Int => ZIO[Any, Throwable, Unit],
    epochs: Int,
    saveEvery: Int
  ): ZIO[Any, Throwable, FitResult] =
    if saveEvery <= 0 then
      ZIO.fail(IllegalArgumentException(s"saveEvery must be > 0, got $saveEvery"))
    else
      def loop(completed: Int, lastResult: Option[FitResult]): ZIO[Any, Throwable, FitResult] =
        val remaining = epochs - completed
        if remaining <= 0 then ZIO.succeed(lastResult.getOrElse(FitResult(Double.NaN, 0)))
        else
          val chunk = math.min(saveEvery, remaining)
          fitChunk(chunk).flatMap { result =>
            val aggregate = result.copy(epochs = completed + result.epochs)
            saveCheckpoint(aggregate.epochs) *>
              ZIO.logInfo(s"Checkpoint saved at epoch ${aggregate.epochs}") *>
              loop(aggregate.epochs, Some(aggregate))
          }
      loop(0, None)

  extension (model: ZModel)
    /** RNN single time-step forward pass (maintains internal state between calls). */
    def rnnTimeStepZ(input: INDArray): Task[INDArray] =
      ZIO.attemptBlocking(model.rnnTimeStep(input).get)

    /** Reset recurrent layer internal state for new independent sequence. */
    def rnnClearPreviousStateZ(): Task[Unit] =
      ZIO.attemptBlocking(model.rnnClearPreviousState().get)

    def predictZ(features: Array[Array[Float]]): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predict(features).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Array[Float]], epochs: Int, lr: Float): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def fitZ(features: Array[Array[Float]], labels: Array[Int], epochs: Int, lr: Float): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(features, labels, epochs, lr).get)

    def predictDoubleZ(features: Array[Array[Double]]): Task[Array[Double]] =
      ZIO.attemptBlocking { val f = features.map(_.map(_.toFloat)); model.predict(f).get.map(_.toDouble) }

    def fitDoubleZ(features: Array[Array[Double]], labels: Array[Double], epochs: Int, lr: Double = 0.001): Task[FitResult] =
      ZIO.attemptBlocking { val f = features.map(_.map(_.toFloat)); val l = labels.map(_.toFloat); model.fit(f, l, epochs, lr.toFloat).get }

    def predictFlow: ZPipeline[Any, Throwable, Array[Array[Float]], Array[Float]] =
      ZPipeline.mapZIO(features => predictZ(features))

    def fitFlow(epochs: Int = 1, lr: Float = 0.001f): ZPipeline[Any, Throwable, (Array[Array[Float]], Array[Float]), FitResult] =
      ZPipeline.mapZIO((feats, labels) => fitZ(feats, labels, epochs, lr))

    def predictTimed(features: Array[Array[Float]]): ZIO[Any, Throwable, Array[Float]] =
      predictZ(features).timed.flatMap((duration, result) => ZIO.logDebug(s"predict: ${duration.toMillis}ms").as(result))

    def fitTimed(features: Array[Array[Float]], labels: Array[Float], epochs: Int, lr: Float = 0.001f): ZIO[Any, Throwable, FitResult] =
      fitZ(features, labels, epochs, lr).timed.flatMap((duration, result) => ZIO.logInfo(s"fit($epochs epochs): ${duration.toMillis}ms, loss=${result.loss}").as(result))

    def fitWithCheckpoints(features: Array[Array[Float]], labels: Array[Float], epochs: Int, saveEvery: Int, checkpointPath: String, lr: Float = 0.001f): ZIO[Any, Throwable, FitResult] =
      if epochs <= 0 then fitZ(features, labels, 0, lr)
      else
        checkpointLoop(
          fitChunk = fitZ(features, labels, _, lr),
          saveCheckpoint = epoch => ZIO.attemptBlocking(model.save(new File(s"$checkpointPath-epoch$epoch")).get),
          epochs = epochs,
          saveEvery = saveEvery
        )

    def evaluateZ(features: Array[Array[Float]], labels: Array[Float], metrics: List[zio.nn.EvalMetric]): Task[Map[String, Double]] =
      ZIO.attemptBlocking(model.evaluate(features, labels, metrics).get)

    /** Train using a DataSetIterator (streaming batches) for N epochs.
      *
      * ZIO-ified version of [[zio.nn.dl4j.wrappers.ZModel.fit(iterator, epochs, lr)]].
      * The iterator is `reset()` at the start of each epoch automatically.
      *
      * @param iterator
      *   A DL4J DataSetIterator (e.g. `MnistDataSetIterator`).
      * @param epochs
      *   Number of training epochs.
      * @param lr
      *   Learning rate (default 0.001).
      * @return
      *   `Task[FitResult]` with loss history.
      *
      * @example {{{
      *   for {
      *     result <- model.fitIteratorZ(iterator, epochs = 5, lr = 0.001)
      *     _      <- ZIO.logInfo(s"Final loss: ${result.loss}")
      *   } yield result
      * }}}
      */
    def fitIteratorZ(iterator: DataSetIterator, epochs: Int, lr: Float = 0.001f): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(iterator, epochs, lr).get)

    /** Evaluate using a DataSetIterator (streaming batches).
      *
      * ZIO-ified version of [[zio.nn.dl4j.wrappers.ZModel.evaluate(iterator, metrics)]].
      * The iterator is `reset()` before iteration.
      *
      * @param iterator
      *   A DL4J DataSetIterator.
      * @param metrics
      *   List of [[zio.nn.EvalMetric]] to compute.
      * @return
      *   `Task[Map[String, Double]]` — metric name → value.
      *
      * @example {{{
      *   metrics <- model.evaluateIteratorZ(testIter, List(EvalMetric.Accuracy))
      *   // Map("accuracy" -> 0.95)
      * }}}
      */
    def evaluateIteratorZ(iterator: DataSetIterator, metrics: List[zio.nn.EvalMetric]): Task[Map[String, Double]] =
      ZIO.attemptBlocking(model.evaluate(iterator, metrics).get)

    def fitZ(iterator: DataSetIterator, epochs: Int): Task[FitResult] =
      ZIO.attemptBlocking(model.fit(iterator, epochs).get)

    def evaluateZ(iterator: DataSetIterator): Task[Map[String, Double]] =
      ZIO.attemptBlocking(model.evaluate(iterator).get)

    def toOnnxZ(path: java.nio.file.Path): Task[Unit] =
      ZIO.attemptBlocking(model.toOnnx(path).get)

    def predictAndStoreZ(
      features: Array[Array[Float]],
      store: zio.nn.VectorStore,
      ids: Array[String]
    ): Task[Array[Float]] =
      ZIO.attemptBlocking(model.predictAndStore(features, store, ids).get)

    def predictAndStoreFlow(
      store: zio.nn.VectorStore
    ): ZPipeline[Any, Throwable, (Array[Array[Float]], Array[String]), Array[Float]] =
      ZPipeline.mapZIO { (features, ids) =>
        predictAndStoreZ(features, store, ids)
      }

    def resumeFromCheckpoint(checkpointPath: String): ZIO[Scope, Throwable, (ZModel, zio.nn.TrainingCheckpoint)] =
      listCheckpoints(checkpointPath).flatMap {
        case Nil => ZIO.fail(new RuntimeException(s"No checkpoints found in $checkpointPath"))
        case checkpoints =>
          val latest = checkpoints.maxBy(_.epoch)
          ZIO.acquireRelease(ZIO.attemptBlocking(ZModel.load(new java.io.File(latest.modelPath)).get))(m => ZIO.attemptBlocking(m.close()).orDie)
            .map((_, latest))
      }

    def loadCheckpoint(path: String): Task[zio.nn.TrainingCheckpoint] =
      ZIO.attemptBlocking {
        val f = new java.io.File(path)
        val epoch = f.getName match
          case s"$prefix-epoch$n" => n.toInt
          case other => throw RuntimeException(s"Cannot parse epoch from filename: $other")
        zio.nn.TrainingCheckpoint(epoch = epoch, modelPath = path, timestamp = f.lastModified())
      }

  // ── Advanced Training: Callbacks, Validation, LR Schedule ──────────────
  extension (model: ZModel)

    /** Train with per-epoch callbacks, early stopping, and LR scheduling.
      *
      * Runs epochs one at a time, firing [[TrainingEvent]]s to all callbacks
      * after each epoch. When validation data is provided, validation loss is
      * computed after each training epoch via forward pass scoring.
      *
      * {{{
      *   val earlyStop = EarlyStopping(patience = 3, minDelta = 0.001)
      *   val cosine    = LRSchedule.cosine(minLr = 0.0001f, maxLr = 0.01f, 10)
      *   model.fitWithCallbacksZ(feats, labels, 50,
      *     callbacks    = List(earlyStop),
      *     lrSchedule   = cosine,
      *     validationData = Some((vFeats, vLabels)))
      * }}}
      */
    def fitWithCallbacksZ(
      features: Array[Array[Float]],
      labels: Array[Float],
      epochs: Int,
      lr: Float = 0.001f,
      callbacks: List[TrainingCallback] = Nil,
      lrSchedule: (Int, Float) => Float = LRSchedule.fixed,
      validationData: Option[(Array[Array[Float]], Array[Float])] = None
    ): ZIO[Any, Throwable, FitResult] =
      def fire(event: TrainingEvent): UIO[Unit] =
        ZIO.foreachDiscard(callbacks)(_.onEvent(event))

      val validationDs: Option[org.nd4j.linalg.dataset.DataSet] = validationData.map { (vf, vl) =>
        new org.nd4j.linalg.dataset.DataSet(Nd4j.create(vf), Nd4j.create(vl.map(Array(_))))
      }

      def isStopped: Boolean = callbacks.exists {
        case es: EarlyStopping => es.shouldStop
        case _ => false
      }

      def loop(epoch: Int, history: List[Double], currentLr: Float, lastValLoss: Option[Double]): ZIO[Any, Throwable, FitResult] =
        if epoch > epochs then
          val result = FitResult(history.lastOption.getOrElse(Double.NaN), epochs, history, validationLoss = lastValLoss)
          fire(TrainingEvent.TrainEnd(result)).as(result)
        else
          for
            start       <- Clock.nanoTime
            epochResult <- ZIO.attemptBlocking(model.fit(features, labels, 1, currentLr).get)
            elapsed     <- Clock.nanoTime.map(ns => (ns - start) / 1000000)
            loss         = epochResult.loss
            _           <- fire(TrainingEvent.EpochEnd(epoch, loss, elapsed))
            newValLoss  <- ZIO.foldLeft(validationDs)(lastValLoss) { (_, vDs) =>
                             ZIO.attemptBlocking(model.underlying.score(vDs)).flatMap { vLoss =>
                               fire(TrainingEvent.ValidationEnd(epoch, vLoss)).as(Some(vLoss))
                             }
                           }
            nextLr       = lrSchedule(epoch, currentLr)
            nextHistory  = history :+ loss
            result      <-
              if isStopped then
                val fr = FitResult(loss, epoch, nextHistory, validationLoss = newValLoss)
                fire(TrainingEvent.TrainEnd(fr)).as(fr)
              else
                loop(epoch + 1, nextHistory, nextLr, newValLoss)
          yield result

      loop(1, Nil, lr, None)

    /** Train with automatic validation split and early stopping.
      *
      * Splits data into training and validation sets, trains with
      * [[EarlyStopping]] enabled, and returns [[FitResult]] with
      * validation loss.
      */
    def fitWithValidationZ(
      features: Array[Array[Float]],
      labels: Array[Float],
      epochs: Int,
      lr: Float = 0.001f,
      validationSplit: Double = 0.2,
      patience: Int = 5,
      callbacks: List[TrainingCallback] = Nil
    ): ZIO[Any, Throwable, FitResult] =
      val splitIdx = (features.length * (1.0 - validationSplit)).toInt.max(1)
      val (trainFeats, valFeats) = features.splitAt(splitIdx)
      val (trainLabels, valLabels) = labels.splitAt(splitIdx)
      val earlyStop = EarlyStopping(patience = patience)
      fitWithCallbacksZ(
        features = trainFeats,
        labels = trainLabels,
        epochs = epochs,
        lr = lr,
        callbacks = earlyStop :: callbacks,
        validationData = Some((valFeats, valLabels))
      )

  // ── ZTokenizer ZIO wrappers ────────────────────────────────────────────
  extension (tok: ZTokenizer)
    def encodeZ(text: String): Task[EncodingResult] =
      ZIO.attemptBlocking(tok.encode(text).get)

    def batchEncodeZ(texts: Array[String]): Task[Array[EncodingResult]] =
      ZIO.attemptBlocking(tok.batchEncode(texts).get)

    def decodeZ(tokens: Array[Int]): Task[String] =
      ZIO.attemptBlocking(tok.decode(tokens).get)

  def regexTokenizer(
    pattern: String = "\\W+",
    config: zio.nn.TokenizerConfig = zio.nn.TokenizerConfig()
  ): ZIO[Scope, Throwable, ZTokenizer] =
    ZIO.acquireRelease(ZIO.attemptBlocking(ZTokenizer.regex(pattern, config).get))(m => ZIO.attemptBlocking(m.close()).orDie)

  def whitespaceTokenizer(
    config: zio.nn.TokenizerConfig = zio.nn.TokenizerConfig()
  ): ZTokenizer = ZTokenizer.whitespace(config)

  def create(arch: zio.nn.ModelDef): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(ZIO.attemptBlocking(ZModel.create(arch)))(m => ZIO.attemptBlocking(m.close()).orDie)

  def load(file: File): ZIO[Scope, Throwable, ZModel] =
    ZIO.acquireRelease(ZIO.attemptBlocking(ZModel.load(file).get))(m => ZIO.attemptBlocking(m.close()).orDie)

  // ── Checkpoint utilities ────────────────────────────────────────────────

  def listCheckpoints(dir: String): Task[List[zio.nn.TrainingCheckpoint]] =
    ZIO.attemptBlocking {
      val d = new java.io.File(dir)
      if !d.isDirectory then throw new RuntimeException(s"Not a directory: $dir")
      val files = d.listFiles().filter(f => f.getName.matches(".+-epoch\\d+$")).toList
      val cps = files.map { f =>
        val n = f.getName.replaceAll("^.*-epoch", "").toInt
        zio.nn.TrainingCheckpoint(epoch = n, modelPath = f.getAbsolutePath, timestamp = f.lastModified())
      }
      cps.sortBy(_.epoch)
    }

  def cleanCheckpoints(dir: String, keep: Int): Task[Unit] =
    listCheckpoints(dir).flatMap { cps =>
      val toDelete = cps.dropRight(keep)
      ZIO.foreachDiscard(toDelete) { cp =>
        ZIO.attemptBlocking { val f = new java.io.File(cp.modelPath); if f.exists() then f.delete() }
      }
    }
