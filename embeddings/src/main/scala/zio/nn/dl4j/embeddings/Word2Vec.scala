package zio.nn.dl4j.embeddings

import zio.*
import zio.stream.*
import zio.nn.*
import zio.nn.dl4j.ZTokenizer
import zio.nn.dsl.*
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.word2vec.VocabWord
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import scala.jdk.CollectionConverters.*
import scala.util.Try

case class Word2VecModel(underlying: SequenceVectors[VocabWord]):

  def vocabSize: Int = underlying.getVocab.numWords()

  def dim: Int = underlying.getLayerSize

  /** Expose the underlying DL4J WordVectors for backward compatibility with
    * legacy DL4J data pipelines (e.g. DataSetIteratorWord2Vec) that expect
    * a raw [[org.deeplearning4j.models.embeddings.wordvectors.WordVectors]].
    *
    * {{{
    *   val w2v: Word2VecModel = Word2Vec.loadGoogleNewsVectors(path).get
    *   val raw: WordVectors    = w2v.vectors
    *   val iter = new DataSetIteratorWord2Vec(dataDir, raw, batch, trunc, train)
    * }}}
    */
  def vectors: WordVectors = underlying

  def similarity(word1: String, word2: String): Task[Double] =
    ZIO.attempt(underlying.similarity(word1, word2))

  def wordsNearest(word: String, n: Int): Task[List[String]] =
    ZIO.attempt(underlying.wordsNearest(word, n).asScala.toList)

  def save(path: Path): Task[Unit] =
    ZIO.attemptBlocking {
      import org.deeplearning4j.models.word2vec.Word2Vec as DL4JWord2Vec
      val w2v = underlying.asInstanceOf[DL4JWord2Vec]
      WordVectorSerializer.writeWord2VecModel(w2v, path.toFile)
    }

  /** Build a dense word→index vocabulary from the model's vocab cache.
    * Uses `words()` (collection-based) instead of `wordAtIndex()` (index-based)
    * because DL4J's VocabCache may have gaps in index-based iteration.
    */
  private def buildVocabMap: Map[String, Int] =
    underlying.getVocab.words().asScala.toArray
      .map(w => w -> underlying.getVocab.indexOf(w))
      .toMap

  def toEmbeddingWeights: Try[EmbeddingWeights] = Try {
    val vocabMap = buildVocabMap
    val vectors = (0 until vocabSize).map { i =>
      underlying.getLookupTable.getWeights.getRow(i.toLong).dup().data().asFloat()
    }.toArray
    EmbeddingWeights(vocabMap, vectors)
  }

  def toEmbeddingLayer(): LayerSpec =
    val weights = toEmbeddingWeights.get
    zio.nn.dsl.Embedding(vocabSize, dim, weights)

  /** Create a tokenizer backed by this Word2Vec model's vocabulary.
    * Unknown tokens are mapped to index `vocabSize` (<UNK>).
    * The resulting tokenizer feeds directly into `model.fitInt` / `model.predictInt`.
    *
    * {{{
    *   val w2v    = Word2Vec.load(path).get
    *   val tok    = w2v.toTokenizer()
    *   val ids    = tok.encode("hello world").get.tokenIds
    *   model.fitInt(Array(ids), labels, epochs = 5)
    * }}}
    */
  def toTokenizer(config: TokenizerConfig = TokenizerConfig()): ZTokenizer =
    val vocab = buildVocabMap
    ZTokenizer.fromVocabulary(
      vocabulary = vocab,
      config     = config,
      unkIndex   = Some(vocabSize)
    )

object Word2Vec:

  enum Algorithm:
    case SkipGram, CBOW

  case class Config(
    dimensions: Int = 300,
    minWordFrequency: Int = 5,
    windowSize: Int = 5,
    batchSize: Int = 512,
    epochs: Int = 3,
    algorithm: Algorithm = Algorithm.SkipGram
  )

  /** Train a Word2Vec model from a stream of sentences.
    *
    * The corpus is collected, written to a temporary file, and fed to DL4J's
    * Word2Vec trainer (SkipGram or CBOW). The trained model is returned as a
    * [[Word2VecModel]] which supports similarity queries, save/load, and
    * conversion to zio-nn embedding layers.
    *
    * {{{
    *   val corpus: ZStream[Any, Throwable, String] = ZStream("the cat sat on the mat", "dogs are great")
    *   ZIO.scoped {
    *     Word2Vec.train(corpus, Config(dimensions = 50, epochs = 1)).flatMap { w2v =>
    *       w2v.similarity("cat", "mat")
    *     }
    *   }
    * }}}
    *
    * @param corpus  stream of sentences (each element is one sentence)
    * @param config  training configuration
    */
  def train(corpus: ZStream[Any, Throwable, String], config: Config): ZIO[Scope, Throwable, Word2VecModel] =
    ZIO.scoped {
      for
        sentences <- corpus.runCollect
        tmpFile   <- ZIO.attempt(java.nio.file.Files.createTempFile("w2v-corpus-", ".txt"))
        _         <- ZIO.addFinalizer(
                       ZIO.attempt(java.nio.file.Files.deleteIfExists(tmpFile)).ignore
                     )
        _         <- ZIO.attempt(
                       java.nio.file.Files.write(tmpFile, sentences.mkString("\n").getBytes(StandardCharsets.UTF_8))
                     )
        model     <- trainFromFile(tmpFile, config)
      yield model
    }

  private def trainFromFile(corpusFile: java.nio.file.Path, config: Config): ZIO[Any, Throwable, Word2VecModel] =
    ZIO.attemptBlockingInterrupt {
      import org.deeplearning4j.models.word2vec.Word2Vec as DL4JWord2Vec
      import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
      import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
      import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
      import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram
      import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW

      val iter = new BasicLineIterator(corpusFile.toFile)
      val tokenizerFactory = new DefaultTokenizerFactory()
      tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor())

      val learningAlgo = config.algorithm match
        case Algorithm.SkipGram => new SkipGram[VocabWord]()
        case Algorithm.CBOW     => new CBOW[VocabWord]()

      val w2v = DL4JWord2Vec.Builder()
        .minWordFrequency(config.minWordFrequency)
        .batchSize(config.batchSize)
        .epochs(config.epochs)
        .layerSize(config.dimensions)
        .windowSize(config.windowSize)
        .iterate(iter)
        .tokenizerFactory(tokenizerFactory)
        .elementsLearningAlgorithm(learningAlgo)
        .build()

      w2v.fit()
      Word2VecModel(w2v)
    }

  def load(path: Path): Try[Word2VecModel] = Try {
    import org.deeplearning4j.models.word2vec.Word2Vec as DL4JWord2Vec
    // Try readWord2VecModel first (format used by save()),
    // fall back to loadStaticModel for pre-existing .bin files
    val vecs =
      try WordVectorSerializer.readWord2VecModel(path.toFile)
      catch case _: Exception =>
        WordVectorSerializer.loadStaticModel(path.toFile)
    vecs match
      case sv: SequenceVectors[?] => Word2VecModel(sv.asInstanceOf[SequenceVectors[VocabWord]])
      case other =>
        throw new UnsupportedOperationException(
          s"Loaded model is ${other.getClass.getSimpleName}, expected SequenceVectors. " +
          s"Try Word2Vec.loadGoogleNewsVectors() for .bin files.")
  }

  def loadGoogleNewsVectors(path: Path): Try[Word2VecModel] = Try {
    val vecs = WordVectorSerializer.readWord2VecModel(path.toFile)
    Word2VecModel(vecs)
  }

  def loadGloVe(path: Path): Try[Word2VecModel] = Try {
    val vecs = WordVectorSerializer.loadTxtVectors(path.toFile)
    vecs match
      case sv: SequenceVectors[?] => Word2VecModel(sv.asInstanceOf[SequenceVectors[VocabWord]])
      case _ => throw new UnsupportedOperationException(
        s"GloVe loader returned ${vecs.getClass.getName}, expected SequenceVectors[VocabWord].")
  }
