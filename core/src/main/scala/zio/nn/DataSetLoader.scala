package zio.nn

import zio.*
import zio.stream.*
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.util.Try

/** ZIO Stream-based batch data loader.
  *
  * Chains file discovery → byte reading → transform → batching into a
  * [[ZStream]] of `(features, labels)` tuples consumable by
  * [[zio.nn.zioApi.fitFlow]] / [[zio.nn.zioApi.predictFlow]].
  *
  * Backend-agnostic: the transform function bridges to backend-specific
  * ImageTransformer or ZTokenizer, so the same pipeline works with both DL4J and DJL.
  *
  * == Image classification with DL4J ==
  * {{{
  *   import zio.nn.dl4j.ImageTransformer
  *   val pipeline = ImagePipeline(Resize(28, 28), Normalize(mean, std))
  *   DataSetLoader.fromImageDir(Path.of("data/train"), pipeline, batchSize = 64) {
  *     (bytes, pl) => ImageTransformer(pl).transform(bytes)
  *   }.flatMap(_.batches.via(model.fitFlow(epochs = 1)).runCollect)
  * }}}
  *
  * == Text classification with Word2Vec ==
  * {{{
  *   val w2v = Word2Vec.load(path).get
  *   val tok = w2v.toTokenizer()
  *   DataSetLoader.fromTextDir(Path.of("data/imdb/train"), batchSize = 32) {
  *     (text, _) => tok.encode(text).map(_.tokenIds)
  *   }.flatMap(_.batches.via(model.fitFlow).runCollect)
  * }}}
  */
case class DataSetLoader(batches: ZStream[Any, Throwable, (Array[Array[Float]], Array[Float])])

object DataSetLoader:

  trait LabelExtractor:
    def apply(path: Path): Try[Float]

  object LabelExtractor:

    def fromParentDir(rootDir: Path): LabelExtractor =
      val classNames = Files.newDirectoryStream(rootDir).asScala
        .filter(Files.isDirectory(_))
        .map(_.getFileName.toString).toSeq.sorted.zipWithIndex.toMap
      (path: Path) => Try {
        val dir = path.getParent.getFileName.toString
        classNames.getOrElse(dir, throw new IllegalArgumentException(s"Unknown class: $dir"))
      }.map(_.toFloat)

    def constant(value: Float): LabelExtractor = (_: Path) => Try(value)

  def fromImageDir(
    rootDir: Path,
    pipeline: ImagePipeline,
    batchSize: Int = 32,
    labelExtr: LabelExtractor = null,
    extensions: Set[String] = Set(".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    parallelism: Int = 4
  )(
    transform: (Array[Byte], ImagePipeline) => Try[Array[Array[Float]]]
  ): ZIO[Any, Throwable, DataSetLoader] =
    val extractor = if labelExtr == null then LabelExtractor.fromParentDir(rootDir) else labelExtr
    fromFiles(rootDir, extensions, extractor, batchSize, parallelism) { (bytes, _) =>
      transform(bytes, pipeline).map(_.flatten)
    }

  def fromTextDir(
    rootDir: Path,
    batchSize: Int = 32,
    labelExtr: LabelExtractor = null,
    parallelism: Int = 4
  )(
    transform: (String, Unit) => Try[Array[Int]]
  ): ZIO[Any, Throwable, DataSetLoader] =
    val extractor = if labelExtr == null then LabelExtractor.fromParentDir(rootDir) else labelExtr
    fromFiles(rootDir, Set(".txt", ".csv", ".json", ".md"), extractor, batchSize, parallelism) {
      (bytes, _) => transform(new String(bytes, java.nio.charset.StandardCharsets.UTF_8), ()).map(_.map(_.toFloat))
    }

  def fromFiles(
    rootDir: Path,
    extensions: Set[String] = Set(".jpg", ".jpeg", ".png"),
    labelExtr: LabelExtractor,
    batchSize: Int = 32,
    parallelism: Int = 4
  )(
    transform: (Array[Byte], Unit) => Try[Array[Float]]
  ): ZIO[Any, Throwable, DataSetLoader] =
    ZIO.attemptBlocking {
      val files = Files.walk(rootDir).iterator().asScala
        .filter(Files.isRegularFile(_))
        .filter(p => extensions.exists(ext => p.getFileName.toString.toLowerCase.endsWith(ext)))
        .toArray

      val stream: ZStream[Any, Throwable, (Array[Array[Float]], Array[Float])] =
        ZStream.fromIterable(files)
          .mapZIOPar(parallelism) { path =>
            for
              bytes <- ZIO.attemptBlocking(Files.readAllBytes(path))
              label <- ZIO.fromTry(labelExtr(path))
              feats <- ZIO.fromTry(transform(bytes, ()))
            yield (feats, label)
          }
          .grouped(batchSize)
          .map { chunk =>
            val (featList, labelList) = chunk.toList.unzip
            val maxLen = if featList.isEmpty then 0 else featList.map(_.length).max
            val paddedFeats = featList.map { f =>
              if f.length < maxLen then f ++ Array.fill(maxLen - f.length)(0.0f) else f
            }.toArray
            (paddedFeats, labelList.toArray)
          }

      DataSetLoader(stream)
    }
