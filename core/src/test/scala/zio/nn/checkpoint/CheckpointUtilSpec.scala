package zio.nn.checkpoint

import zio.*
import zio.test.*
import zio.test.Assertion.*
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

object CheckpointUtilSpec extends ZIOSpecDefault:

  private val testDir = "target/checkpoint-test"

  def spec = suite("CheckpointUtil")(
    // ── checkpointPath ──
    suite("checkpointPath")(
      test("generates correct path") {
        val path = CheckpointUtil.checkpointPath("/tmp/ckpts", 10, 0)
        assertTrue(path == "/tmp/ckpts/checkpoint_e10_r0.json")
      },
      test("handles trailing slash in dir") {
        val path = CheckpointUtil.checkpointPath("/tmp/ckpts/", 1, 2)
        assertTrue(path == "/tmp/ckpts/checkpoint_e1_r2.json")
      },
      test("default round is 0") {
        val path = CheckpointUtil.checkpointPath("/tmp/ckpts", 5)
        assertTrue(path == "/tmp/ckpts/checkpoint_e5_r0.json")
      }
    ),

    // ── save / load roundtrip ──
    suite("save / load")(
      test("save and load a ParameterCheckpoint") {
        val ckpt = ParameterCheckpoint(
          epoch = 10,
          round = 1,
          parametersJson = """{"w1":[1.0,2.0],"b1":[0.0]}""",
          metadata = Map("loss" -> "0.123", "accuracy" -> "0.95")
        )
        val path = testDir + "/roundtrip.json"

        for
          _       <- CheckpointUtil.save(path, ckpt)
          loaded  <- CheckpointUtil.load(path)
        yield
          val exists = Files.exists(Paths.get(path))
          assertTrue(
            exists,
            loaded.isDefined,
            loaded.get.epoch == 10,
            loaded.get.round == 1,
            loaded.get.parametersJson == ckpt.parametersJson,
            loaded.get.metadata.get("loss").contains("0.123"),
            loaded.get.metadata.get("accuracy").contains("0.95")
          )
      },
      test("load non-existent file returns None") {
        val result = CheckpointUtil.load("/nonexistent/path.json")
        assertZIO(result)(isNone)
      },
      test("load corrupted file returns None") {
        val path = testDir + "/corrupt.json"
        for
          _      <- ZIO.attemptBlocking(Files.writeString(Paths.get(path), "not valid json"))
          loaded <- CheckpointUtil.load(path)
        yield assertTrue(loaded.isEmpty)
      },
      test("multiple checkpoints are independent") {
        val ckpt1 = ParameterCheckpoint(epoch = 1, parametersJson = "{}")
        val ckpt2 = ParameterCheckpoint(epoch = 2, parametersJson = "{}")

        for
          _ <- CheckpointUtil.save(testDir + "/ckpt1.json", ckpt1)
          _ <- CheckpointUtil.save(testDir + "/ckpt2.json", ckpt2)
          l1 <- CheckpointUtil.load(testDir + "/ckpt1.json")
          l2 <- CheckpointUtil.load(testDir + "/ckpt2.json")
        yield assertTrue(
          l1.get.epoch == 1,
          l2.get.epoch == 2
        )
      }
    ),

    // ── cleanup ──
    suite("cleanup")(
      test("removes all files when keep=0") {
        val dir = testDir + "/remove-all"
        for
          _ <- ZIO.attempt(Files.createDirectories(Paths.get(dir)))
          _ <- CheckpointUtil.save(s"$dir/model_e1.json", ParameterCheckpoint(1, parametersJson = "{}"))
          _ <- CheckpointUtil.save(s"$dir/model_e2.json", ParameterCheckpoint(2, parametersJson = "{}"))
          _ <- CheckpointUtil.cleanup(dir, "model", keep = 0)
          remaining <- ZIO.attempt {
            Files.list(Paths.get(dir)).iterator().asScala
              .filter(_.getFileName.toString.startsWith("model"))
              .toList
          }
        yield assertTrue(remaining.isEmpty)
      },
      test("keeps all files when keep >= count") {
        val dir = testDir + "/keep-all"
        for
          _ <- ZIO.attempt(Files.createDirectories(Paths.get(dir)))
          _ <- CheckpointUtil.save(s"$dir/model_e1.json", ParameterCheckpoint(1, parametersJson = "{}"))
          _ <- CheckpointUtil.save(s"$dir/model_e2.json", ParameterCheckpoint(2, parametersJson = "{}"))
          _ <- CheckpointUtil.cleanup(dir, "model", keep = 5)
          remaining <- ZIO.attempt {
            Files.list(Paths.get(dir)).iterator().asScala
              .filter(_.getFileName.toString.startsWith("model"))
              .toList.size
          }
        yield assertTrue(remaining == 2)
      },
      test("cleanup ignores non-matching files") {
        val dir = testDir + "/ignore-nonmatch"
        for
          _ <- ZIO.attempt(Files.createDirectories(Paths.get(dir)))
          _ <- CheckpointUtil.save(s"$dir/model_e1.json", ParameterCheckpoint(1, parametersJson = "{}"))
          _ <- ZIO.attempt(Files.writeString(Paths.get(s"$dir/other.txt"), "data"))
          _ <- CheckpointUtil.cleanup(dir, "model", keep = 0)
          remaining <- ZIO.attempt {
            Files.list(Paths.get(dir)).iterator().asScala.toList
          }
        yield assertTrue(remaining.size == 1, remaining.head.getFileName.toString == "other.txt")
      },
      test("cleanup on non-existent dir is a no-op") {
        val result = CheckpointUtil.cleanup("/nonexistent/dir", "prefix", 3)
        assertZIO(result)(isUnit)
      }
    )
  )
