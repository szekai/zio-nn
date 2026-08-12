package zio.nn.xgboost

import zio.nn.*
import zio.nn.dsl.*
import zio.test.*
import zio.test.Assertion.*

object XGBoostSpec extends ZIOSpecDefault:

  private def synthData(n: Int, dim: Int = 2): (Array[Array[Float]], Array[Float]) =
    val rng = new scala.util.Random(7)
    val features = Array.tabulate(n)(_ => Array.fill(dim)(rng.nextFloat() * 4f - 2f))
    val labels = features.map(f => 3f * f(0) - 2f * f(1) + 0.5f)
    (features, labels)

  private val arch: ModelDef =
    Sequential(2)(Dense(16, ReLU), Output(1, MSE)).build

  private def pearson(xs: Array[Float], ys: Array[Float]): Double =
    val n = xs.length
    val mx = xs.sum / n
    val my = ys.sum / n
    var cov = 0.0
    var vx = 0.0
    var vy = 0.0
    xs.zip(ys).foreach { (x, y) =>
      val dx = x - mx
      val dy = y - my
      cov += dx * dy
      vx += dx * dx
      vy += dy * dy
    }
    cov / math.sqrt(vx * vy)

  def spec = suite("XGBoost backend")(
    test("fit trains a regression model approximating y = 3x1 - 2x2 + 0.5") {
      val (feats, labels) = synthData(200)
      val model = ZModel.create(arch).get
      val fitResult = model.fit(feats, labels, epochs = 60, lr = 0.1f).get
      val preds = model.predict(feats).get
      model.close()
      assertTrue(
        fitResult.epochs == 60,
        fitResult.lossHistory.nonEmpty,
        fitResult.loss >= 0.0,
        math.abs(pearson(preds, labels)) > 0.99
      )
    },
    test("predictDirect flattens RNN-shaped input and replicates the prediction") {
      // flat window width = timeSteps x featCount (10 = 5 x 2)
      val (feats, labels) = synthData(200, dim = 10)
      val flatArch = Sequential(10)(Dense(16, ReLU), Output(1, MSE)).build
      val model = ZModel.create(flatArch).get
      model.fit(feats, labels, epochs = 40, lr = 0.1f).get
      val sample = Array.fill(5)(Array(1.0f, 0.5f)) // 5 rows x 2 feats -> 10 flat
      val preds = model.predictDirect(sample, 5, 2).get
      model.close()
      assertTrue(preds.length == 5, preds.forall(_ == preds.head))
    },
    test("fitArray3D flattens (samples x timeSteps x features)") {
      val (feats2D, labels) = synthData(60)
      val feats3D = feats2D.map(Array(_)) // 1 timestep per sample
      val model = ZModel.create(arch).get
      val result = model.fitArray3D(feats3D, labels, epochs = 40, batchSize = 16, lr = 0.1f).get
      model.close()
      assertTrue(result.loss >= 0.0, result.epochs == 40)
    },
    test("save/load round-trip preserves predictions") {
      val (feats, labels) = synthData(200)
      val model = ZModel.create(arch).get
      model.fit(feats, labels, epochs = 40, lr = 0.1f).get
      val before = model.predict(feats).get
      val path = java.nio.file.Path.of("target/xgb-test-model")
      model.save(path).get
      model.close()
      val loaded = ZModel.load(path).get
      val after = loaded.predict(feats).get
      loaded.close()
      assertTrue(
        before.length == after.length,
        before.zip(after).forall { (a, b) => math.abs(a - b) < 1e-3 }
      )
    },
    test("Backend.compile rejects recurrent/unsupported layers") {
      val lstmArch = Sequential(7)(LSTM(32, Tanh), Dense(16, ReLU), Output(1, MSE)).build
      assertTrue(Backend.compile(lstmArch).isFailure)
    },
    test("predict before fit fails cleanly") {
      val model = ZModel.create(arch).get
      val result = model.predict(Array(Array(1f, 2f)))
      model.close()
      assertTrue(result.isFailure)
    }
  )
