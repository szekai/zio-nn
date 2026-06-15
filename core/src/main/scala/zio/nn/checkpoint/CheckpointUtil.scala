package zio.nn.checkpoint

import zio.*
import zio.json.*
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

/**
 * Training checkpoint for crash recovery — stores epoch, round, and serialized parameters.
 *
 * Unlike the existing [[zio.nn.TrainingCheckpoint]] (which tracks a model file path),
 * this stores parameters as an inline JSON blob for fine-grained checkpointing
 * during training loops.
 */
case class ParameterCheckpoint(
  epoch: Int,
  round: Int = 0,
  parametersJson: String,
  metadata: Map[String, String] = Map.empty,
  timestamp: Long = java.lang.System.currentTimeMillis()
) derives JsonCodec

/**
 * Persisted model wrapper — serializes parameters JSON alongside metadata.
 * Supports backwards-compatible load (falls back to params-only format).
 */
case class PersistedModel(
  parameters: String,
  metadata: Map[String, String] = Map.empty,
  modelType: String = ""
) derives JsonCodec

/**
 * ZIO-wrapped utilities for saving/loading/cleaning training checkpoints.
 *
 * All I/O operations are wrapped in [[ZIO.attemptBlocking]] to run on the
 * blocking thread pool, preventing file-system stalls from starving the main
 * ZIO fiber pool.
 */
object CheckpointUtil {

  /**
   * Writes a [[ParameterCheckpoint]] to the given file path as JSON.
   * Creates parent directories if they do not exist.
   */
  def save(path: String, checkpoint: ParameterCheckpoint): Task[Unit] =
    ZIO.attemptBlocking {
      val dir = Paths.get(path).getParent
      if (dir != null && !Files.exists(dir)) Files.createDirectories(dir)
      Files.writeString(Paths.get(path), checkpoint.toJson)
    }

  /**
   * Reads a [[ParameterCheckpoint]] from the given file path.
   * Returns `None` if the file does not exist or fails to deserialize.
   */
  def load(path: String): Task[Option[ParameterCheckpoint]] =
    ZIO.attemptBlocking {
      val p = Paths.get(path)
      if (!Files.exists(p)) None
      else Files.readString(p).fromJson[ParameterCheckpoint] match
        case Right(ckpt) => Some(ckpt)
        case Left(_)     => None
    }

  /**
   * Deletes all checkpoint files in `dir` whose filename starts with `prefix`,
   * keeping only the `keep` most recently modified files.
   */
  def cleanup(dir: String, prefix: String, keep: Int): Task[Unit] =
    ZIO.attemptBlocking {
      val d = Paths.get(dir)
      if (!Files.exists(d)) ()
      else {
        val stream = Files.list(d)
        val buf = new java.util.ArrayList[java.nio.file.Path]()
        try
          val iter = stream.iterator()
          while (iter.hasNext) buf.add(iter.next())
        finally stream.close()

        val files = buf.asScala.toList
          .filter(_.getFileName.toString.startsWith(prefix))
          .sortBy(_.toFile.lastModified())

        val toDelete = files.dropRight(keep)
        toDelete.foreach(f => Files.deleteIfExists(f))
      }
    }

  /**
   * Generates a checkpoint file path for the given directory, epoch, and round.
   *
   * Example: `checkpointPath("/tmp/ckpts", 10, 0)` → `"/tmp/ckpts/checkpoint_e10_r0.json"`
   */
  def checkpointPath(dir: String, epoch: Int, round: Int = 0): String = {
    val d = if (dir.endsWith("/")) dir else dir + "/"
    d + s"checkpoint_e${epoch}_r${round}.json"
  }
}
