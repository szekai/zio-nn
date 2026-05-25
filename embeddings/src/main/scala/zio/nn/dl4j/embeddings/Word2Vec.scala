package zio.nn.dl4j.embeddings

import zio.*
import zio.stream.*
import zio.nn.*
import zio.nn.dsl.*
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.word2vec.VocabWord
import java.nio.file.Path
import scala.jdk.CollectionConverters.*
import scala.util.Try

case class Word2VecModel(underlying: SequenceVectors[VocabWord]):

  def vocabSize: Int = underlying.getVocab.numWords()

  def dim: Int = underlying.getLayerSize

  def similarity(word1: String, word2: String): Task[Double] =
    ZIO.attempt(underlying.similarity(word1, word2))

  def wordsNearest(word: String, n: Int): Task[List[String]] =
    ZIO.attempt(underlying.wordsNearest(word, n).asScala.toList)

  def save(path: Path): Task[Unit] =
    ZIO.attemptBlocking {
      val fos = java.io.FileOutputStream(path.toFile)
      try WordVectorSerializer.writeSequenceVectors(underlying, fos)
      finally fos.close()
    }

  def toEmbeddingWeights: Try[EmbeddingWeights] = Try {
    val words = (0 until vocabSize).map(i => underlying.getVocab.wordFor(i).getWord).toArray
    val vocabMap = words.zipWithIndex.toMap
    val vectors = (0 until vocabSize).map { i =>
      underlying.getLookupTable.getWeights.getRow(i.toLong).dup().data().asFloat()
    }.toArray
    EmbeddingWeights(vocabMap, vectors)
  }

  def toEmbeddingLayer(): LayerSpec =
    val weights = toEmbeddingWeights.get
    zio.nn.dsl.Embedding(vocabSize, dim, weights)

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

  def train(corpus: ZStream[Any, Throwable, String], config: Config): ZIO[Scope, Throwable, Word2VecModel] =
    ZIO.fail(new UnsupportedOperationException(
      "Word2Vec.train() requires DL4J SequenceVectors APIs that are not yet implemented for DL4J 1.0.0-M2.1. " +
      "Use Word2Vec.load() to load pre-trained vectors, or train externally and import the result."))

  def load(path: Path): Try[Word2VecModel] = Try {
    val vecs = WordVectorSerializer.loadStaticModel(path.toFile)
    vecs match
      case sv: SequenceVectors[?] => Word2VecModel(sv.asInstanceOf[SequenceVectors[VocabWord]])
      case _ =>
        val loaded = WordVectorSerializer.readWord2VecModel(path.toFile)
        Word2VecModel(loaded)
  }

  def loadGoogleNewsVectors(path: Path): Try[Word2VecModel] = Try {
    val vecs = WordVectorSerializer.readWord2VecModel(path.toFile)
    Word2VecModel(vecs)
  }

  def loadGloVe(path: Path): Try[org.deeplearning4j.models.embeddings.wordvectors.WordVectors] = Try {
    WordVectorSerializer.loadTxtVectors(path.toFile)
  }
