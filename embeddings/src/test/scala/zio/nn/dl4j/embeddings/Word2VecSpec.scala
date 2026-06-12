package zio.nn.dl4j.embeddings

import zio.*
import zio.stream.*
import zio.test.*
import zio.test.Assertion.*
import zio.nn.*

object Word2VecSpec extends ZIOSpecDefault:

  private val smallCorpus = ZStream(
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
    "the mat was wet",
    "the log was dry",
    "great pets are the best",
    "the cat chased the dog",
    "the dog chased the cat",
    "pets are family",
    "the sun was bright",
    "the moon was bright too",
    "stars shine bright at night"
  )

  private val trainConfig = Word2Vec.Config(
    dimensions = 10,
    minWordFrequency = 1,
    windowSize = 3,
    batchSize = 64,
    epochs = 1
  )

  def spec = suite("Word2Vec (DL4J — embeddings module)")(
    test("train() returns a model with non-zero vocabulary") {
      ZIO.scoped {
        Word2Vec.train(smallCorpus, trainConfig).flatMap { model =>
          ZIO.attempt {
            assertTrue(
              model.vocabSize > 0,
              model.dim == 10,
              model.vectors != null
            )
          }
        }
      }
    },

    test("similarity returns a value in [-1, 1]") {
      ZIO.scoped {
        Word2Vec.train(smallCorpus, trainConfig).flatMap { model =>
          model.similarity("cat", "mat").map { sim =>
            assertTrue(sim >= -1.0, sim <= 1.0)
          }
        }
      }
    },

    test("wordsNearest returns the requested number of words") {
      ZIO.scoped {
        Word2Vec.train(smallCorpus, trainConfig).flatMap { model =>
          model.wordsNearest("cat", 3).map { nearest =>
            assertTrue(nearest.length == 3)
          }
        }
      }
    },

    test("toTokenizer creates a tokenizer backed by Word2Vec vocabulary") {
      ZIO.scoped {
        Word2Vec.train(smallCorpus, trainConfig).map { model =>
          val tok = model.toTokenizer()
          val result = tok.encode("the cat").get
          tok.close()
          assertTrue(
            result.tokenIds.length == 2,
            result.tokenIds.forall(_ < model.vocabSize + 1) // +1 for <UNK>
          )
        }
      }
    },

    test("toTokenizer maps unknown tokens to <UNK> index") {
      ZIO.scoped {
        Word2Vec.train(smallCorpus, trainConfig).map { model =>
          val tok = model.toTokenizer()
          val result = tok.encode("xyznonexistent word").get
          tok.close()
          val unkIndex = model.vocabSize
          assertTrue(
            result.tokenIds.length == 2,
            result.tokenIds.contains(unkIndex)
          )
        }
      }
    },

    test("toEmbeddingWeights returns correct shapes") {
      ZIO.scoped {
        Word2Vec.train(smallCorpus, trainConfig).flatMap { model =>
          ZIO.attempt {
            val weights = model.toEmbeddingWeights.get
            assertTrue(
              weights.vocabulary.size == model.vocabSize,
              weights.vectors.length == model.vocabSize,
              weights.vectors.forall(_.length == model.dim)
            )
          }
        }
      }
    },

    test("save and load round-trip preserves vocabulary") {
      ZIO.scoped {
        for
          model   <- Word2Vec.train(smallCorpus, trainConfig)
          tmpFile <- ZIO.attempt(java.nio.file.Files.createTempFile("w2v-test-", ".bin"))
          _       <- ZIO.addFinalizer(
                       ZIO.attempt(java.nio.file.Files.deleteIfExists(tmpFile)).ignore
                     )
          _       <- model.save(tmpFile)
          loaded  <- ZIO.attemptBlocking(Word2Vec.load(tmpFile).get)
          sim     <- loaded.similarity("cat", "mat")
        yield assertTrue(
          loaded.vocabSize == model.vocabSize,
          loaded.dim == model.dim,
          sim >= -1.0
        )
      }
    }
  )
