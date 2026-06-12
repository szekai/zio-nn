package zio.nn

import zio.*
import zio.test.*
import zio.test.Assertion.*
import java.nio.file.{Files, Path}
import scala.util.Try

object DataSetLoaderSpec extends ZIOSpecDefault:

  private def withTempDir(test: Path => ZIO[Any, Throwable, TestResult]): ZIO[Any, Throwable, TestResult] =
    ZIO.attemptBlocking {
      val dir = Files.createTempDirectory("zionn-ds-test-")
      dir.toFile.deleteOnExit()
      dir
    }.flatMap(test)

  private def createImageFile(dir: Path, subdir: String, name: String): Path =
    val sub = dir.resolve(subdir)
    Files.createDirectories(sub)
    val file = sub.resolve(name)
    // Write a minimal valid PNG (67 bytes)
    val minimalPng: Array[Byte] = Array(
      -119, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
      0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, -112, 79, -53, 116,
      0, 0, 0, 12, 73, 68, 65, 84, 8, -39, 99, -8, -1, -1, -1, 0,
      0, 4, 0, 1, -2, -1, 53, -66, 0, 0, 0, 0, 73, 69, 78, 68, -82, 66, 96, -126
    )
    Files.write(file, minimalPng)
    file

  private def createTextFile(dir: Path, subdir: String, name: String, content: String): Path =
    val sub = dir.resolve(subdir)
    Files.createDirectories(sub)
    val file = sub.resolve(name)
    Files.writeString(file, content)
    file

  def spec = suite("DataSetLoader")(
    test("fromImageDir discovers files in subdirectories") {
      withTempDir { dir =>
        createImageFile(dir, "cat", "a.png")
        createImageFile(dir, "cat", "b.png")
        createImageFile(dir, "dog", "c.png")
        val pipeline = ImagePipeline(ImageTransformDef.Resize(2, 2))
        DataSetLoader.fromImageDir(dir, pipeline, batchSize = 2) { (bytes, pl) =>
          // Return a fixed (1×4) feature array for any valid PNG
          if bytes.length > 50 then Try(Array(Array.fill(4)(1.0f)))
          else Try(Array(Array.fill(4)(0.0f)))
        }.flatMap { loader =>
          loader.batches.runCollect.map { batches =>
            val totalSamples = batches.map(_._1.length).sum
            assertTrue(totalSamples == 3)
          }
        }
      }
    },
    test("fromTextDir discovers and tokenizes text files") {
      withTempDir { dir =>
        createTextFile(dir, "pos", "a.txt", "great movie")
        createTextFile(dir, "pos", "b.txt", "loved it")
        createTextFile(dir, "neg", "c.txt", "terrible film")
        DataSetLoader.fromTextDir(dir, batchSize = 2) { (text, _) =>
          Try(text.split("\\s+").map(_.hashCode).map(_.toFloat.toInt))
        }.flatMap { loader =>
          loader.batches.runCollect.map { batches =>
            val totalSamples = batches.map(_._1.length).sum
            assertTrue(totalSamples == 3)
          }
        }
      }
    },
    test("LabelExtractor.fromParentDir maps class names to indices") {
      withTempDir { dir =>
        createImageFile(dir, "cat", "a.png")
        createImageFile(dir, "dog", "b.png")
        val extractor = DataSetLoader.LabelExtractor.fromParentDir(dir)
        val catLabel = extractor(dir.resolve("cat/a.png"))
        val dogLabel = extractor(dir.resolve("dog/b.png"))
        assertTrue(
          catLabel.isSuccess,
          dogLabel.isSuccess,
          catLabel.get == 0f, // "cat" comes before "dog" alphabetically
          dogLabel.get == 1f
        )
      }
    },
    test("fromFiles with custom transform produces correct batch sizes") {
      withTempDir { dir =>
        createImageFile(dir, "data", "a.png")
        createImageFile(dir, "data", "b.png")
        createImageFile(dir, "data", "c.png")
        DataSetLoader.fromFiles(
          dir.resolve("data"),
          extensions = Set(".png"),
          labelExtr = DataSetLoader.LabelExtractor.constant(0f),
          batchSize = 2
        ) { (bytes, _) =>
          Try(Array.fill(4)(1.0f))
        }.flatMap { loader =>
          loader.batches.runCollect.map { batches =>
            val batchSizes = batches.map(_._1.length)
            val total = batchSizes.sum
            assertTrue(
              total == 3,
              batchSizes.forall(sz => sz == 2 || sz == 1),
              batches.forall { case (feats, labs) => feats.length == labs.length }
            )
          }
        }
      }
    }
  )
