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

  /** Load ModelDef + TrainingParams from HOCON.
    * Parses architecture AND training parameters in one call.
    * Use when you need epochs, learningRate, batchSize alongside architecture.
    *
    * Example HOCON:
    *   federated.nn {
    *     sequential { input-size = 30, layers = [...], optimizer = { adam: { learning-rate = 0.001 } } }
    *     training { epochs = 100, learning-rate = 0.01, batch-size = 32 }
    *   }
    */
  def fromHoconWithTraining(path: String): Task[(ModelDef, Option[TrainingParams])] =
    for
      provider <- ZIO.attempt(ConfigProvider.fromHoconFilePath("application.conf"))
      model    <- provider.nested(path).load(deriveConfig[ModelDef])
      training <- ZIO.attempt {
        val conf = com.typesafe.config.ConfigFactory.load().getConfig(path)
        if conf.hasPath("training") then
          val t = conf.getConfig("training")
          Some(TrainingParams(
            epochs = if t.hasPath("epochs") then t.getInt("epochs") else 100,
            learningRate = if t.hasPath("learning-rate") then t.getDouble("learning-rate") else 0.01,
            batchSize = if t.hasPath("batch-size") then t.getInt("batch-size") else 32
          ))
        else None
      }.catchAll(_ => ZIO.none)
    yield (model, training)

  /** Load from an explicit HOCON string. Useful for testing or inline config. */
  def fromString(hocon: String, path: String = "model"): Task[ModelDef] =
    for
      provider <- ZIO.attempt(ConfigProvider.fromHoconString(hocon))
      config   <- provider.nested(path).load(deriveConfig[ModelDef])
    yield config

  /** Load ModelDef + TrainingParams from HOCON string. */
  def fromStringWithTraining(hocon: String, path: String = "model"): Task[(ModelDef, Option[TrainingParams])] =
    for
      provider <- ZIO.attempt(ConfigProvider.fromHoconString(hocon))
      model    <- provider.nested(path).load(deriveConfig[ModelDef])
      training <- ZIO.attempt {
        val conf = com.typesafe.config.ConfigFactory.parseString(hocon).resolve().getConfig(path)
        if conf.hasPath("training") then
          val t = conf.getConfig("training")
          Some(TrainingParams(
            epochs = if t.hasPath("epochs") then t.getInt("epochs") else 100,
            learningRate = if t.hasPath("learning-rate") then t.getDouble("learning-rate") else 0.01,
            batchSize = if t.hasPath("batch-size") then t.getInt("batch-size") else 32
          ))
        else None
      }.catchAll(_ => ZIO.none)
    yield (model, training)

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
      |  training { epochs = 100, learning-rate = 0.01, batch-size = 32 }
      |}""".stripMargin
