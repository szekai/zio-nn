package zio.nn.dl4j

import zio.*
import zio.stream.*
import zio.nn.{EncodingResult, FitResult}
import java.io.File

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
