package zio.nn

import zio.*
import zio.config.*
import zio.config.typesafe.*
import zio.config.magnolia.*

/** Load model architecture from HOCON/YAML config files.
  *
  * Usage:
  * {{{
  *   // application.conf:
  *   //   model {
  *   //     input-size = 7
  *   //     layers = [
  *   //       { type = lstm, n-out = 64, activation = tanh }
  *   //       { type = dense, n-out = 32, activation = relu }
  *   //       { type = output, n-out = 1, loss = mse }
  *   //     ]
  *   //     optimizer = { type = adam, learning-rate = 0.001 }
  *   //     seed = 42
  *   //   }
  *
  *   val arch: Task[ModelDef] = ConfigLoader.fromHocon("model")
  *   val model = for
  *     a <- arch
  *     m <- ZModel.create(a, "config-model")
  *   yield m
  * }}}
  */
object ConfigLoader:

  /** Load a ModelDef from a HOCON path in application.conf.
    * Example path: "models.lstm-trend", "model", "architectures.cnn"
    */
  def fromHocon(path: String): Task[ModelDef] =
    for
      provider <- ZIO.attempt(ConfigProvider.fromHoconFilePath("application.conf"))
      config   <- provider.nested(path).load(deriveConfig[ModelDef])
    yield config

  /** Load from an explicit HOCON string. Useful for testing or inline config. */
  def fromString(hocon: String, path: String = "model"): Task[ModelDef] =
    for
      provider <- ZIO.attempt(ConfigProvider.fromHoconString(hocon))
      config   <- provider.nested(path).load(deriveConfig[ModelDef])
    yield config

  /** Generate the default HOCON template for the default LSTM architecture. */
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
      |}""".stripMargin
