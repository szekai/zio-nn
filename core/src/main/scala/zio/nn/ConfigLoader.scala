package zio.nn
import zio.*

object ConfigLoader:
  def fromHocon(path: String): Task[ModelDef] =
    ZIO.fail(new UnsupportedOperationException("HOCON config temporarily disabled while AdvancedLayerDef is being stabilized. Use the Scala DSL."))
  def fromHoconWithTraining(path: String): Task[(ModelDef, Option[TrainingParams])] =
    ZIO.fail(new UnsupportedOperationException("HOCON config temporarily disabled."))
  def fromString(hocon: String, path: String = "model"): Task[ModelDef] =
    ZIO.fail(new UnsupportedOperationException("HOCON config temporarily disabled."))
  def fromStringWithTraining(hocon: String, path: String = "model"): Task[(ModelDef, Option[TrainingParams])] =
    ZIO.fail(new UnsupportedOperationException("HOCON config temporarily disabled."))
  def defaultHocon: String =
    """model {
      |  sequential {
      |    input-size = 7
      |    layers = [
      |      { type = lstm, n-in = 7, n-out = 64, activation = tanh }
      |      { type = dense, n-in = 64, n-out = 32, activation = relu }
      |      { type = output, n-in = 32, n-out = 1, loss = mse, activation = identity }
      |    ]
      |    optimizer = { adam: { learning-rate = 0.001 } }
      |    seed = 42
      |  }
      |  training { epochs = 100, learning-rate = 0.01, batch-size = 32 }
      |}""".stripMargin
