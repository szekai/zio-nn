package zio.nn
import zio.*
import scala.jdk.CollectionConverters.*

object ConfigLoader:
  private def parseActivation(s: String): ActivationFn =
    ActivationFn.values.find(_.toString.equalsIgnoreCase(s))
      .getOrElse(throw IllegalArgumentException(s"Unknown activation: $s (valid: ${ActivationFn.values.mkString(", ")})"))

  private def parseLoss(s: String): LossFn =
    LossFn.values.find(_.toString.equalsIgnoreCase(s))
      .getOrElse(throw IllegalArgumentException(s"Unknown loss: $s (valid: ${LossFn.values.mkString(", ")})"))

  private def parseLayer(conf: com.typesafe.config.Config): LayerDef =
    conf.getString("type") match
      case "lstm" => LayerDef.LSTM(-1,
        conf.getInt("n-out"),
        parseActivation(conf.getString("activation")),
        if conf.hasPath("dropout") then conf.getDouble("dropout") else 0.0)
      case "dense" => LayerDef.Dense(-1,
        conf.getInt("n-out"),
        parseActivation(conf.getString("activation")))
      case "output" => LayerDef.Output(-1,
        conf.getInt("n-out"),
        if conf.hasPath("loss") then parseLoss(conf.getString("loss")) else LossFn.MSE,
        if conf.hasPath("activation") then parseActivation(conf.getString("activation"))
        else ActivationFn.Identity)
      case "batchnorm" => LayerDef.BatchNorm(-1)
      case "dropout" => LayerDef.Dropout(conf.getDouble("rate"))
      case "embedding" => LayerDef.Embedding(
        conf.getInt("vocab-size"), conf.getInt("embedding-dim"), None)
      case other => throw IllegalArgumentException(s"Unknown layer type: $other")

  private def parseModel(conf: com.typesafe.config.Config): ModelDef =
    val s = conf.getConfig("sequential")
    val layers = s.getConfigList("layers").asScala.map(parseLayer).toList
    val seed = if s.hasPath("seed") then s.getLong("seed") else 42L
    ModelDef.Sequential(SequentialDef(
      inputSize = s.getInt("input-size"),
      layers = layers.map(AnyLayer.Standard(_)),
      seed = seed, convInput = None))

  def fromHocon(path: String): Task[ModelDef] =
    ZIO.attempt {
      val conf = com.typesafe.config.ConfigFactory.load().getConfig(path)
      parseModel(conf)
    }

  def fromHoconWithTraining(path: String): Task[(ModelDef, Option[TrainingParams])] =
    ZIO.attempt {
      val conf = com.typesafe.config.ConfigFactory.load().getConfig(path)
      val model = parseModel(conf)
      val training = if conf.hasPath("training") then
        val t = conf.getConfig("training")
        Some(TrainingParams(
          epochs = if t.hasPath("epochs") then t.getInt("epochs") else 100,
          learningRate = if t.hasPath("learning-rate") then t.getDouble("learning-rate") else 0.01,
          batchSize = if t.hasPath("batch-size") then t.getInt("batch-size") else 32))
      else None
      (model, training)
    }

  def fromString(hocon: String, path: String = "model"): Task[ModelDef] =
    ZIO.attempt {
      parseModel(com.typesafe.config.ConfigFactory.parseString(hocon).resolve().getConfig(path))
    }

  def fromStringWithTraining(hocon: String, path: String = "model"): Task[(ModelDef, Option[TrainingParams])] =
    ZIO.attempt {
      val conf = com.typesafe.config.ConfigFactory.parseString(hocon).resolve().getConfig(path)
      val model = parseModel(conf)
      val training = if conf.hasPath("training") then
        val t = conf.getConfig("training")
        Some(TrainingParams(
          epochs = if t.hasPath("epochs") then t.getInt("epochs") else 100,
          learningRate = if t.hasPath("learning-rate") then t.getDouble("learning-rate") else 0.01,
          batchSize = if t.hasPath("batch-size") then t.getInt("batch-size") else 32))
      else None
      (model, training)
    }

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
