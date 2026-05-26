package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object EdgeCaseSpec extends ZIOSpecDefault:

  def spec = suite("DL4J Edge Cases")(
    test("close() is idempotent") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      model.close(); model.close()
      assertTrue(true)
    },
    test("predict returns correct number of outputs") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val result = model.predict(Array.fill(4)(Array.fill(7)(0.5f)))
      assertTrue(result.isSuccess && result.get.length == 4)
    },
    test("fit with zero epochs is no-op") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val result = model.fit(Array.fill(5)(Array.fill(7)(0.1f)), Array.fill(5)(0.5f), 0)
      assertTrue(result.isSuccess)
    },
    test("fit then predict doesn't crash") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val feats = Array.fill(10)(Array.fill(7)(0.1f))
      val labels = Array.fill(10)(0.5f)
      model.fit(feats, labels, 2)
      val pred = model.predict(feats)
      assertTrue(pred.isSuccess)
    },
    test("underlying model accessible via escape hatch") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      assertTrue(model.underlying != null)
    },
    test("save then load roundtrip") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val tmp = java.io.File.createTempFile("zionn", ".zip")
      tmp.deleteOnExit()
      model.save(tmp)
      model.close()
      ZModel.load(tmp) match
        case scala.util.Success(loaded) =>
          val pred = loaded.predict(Array.fill(1)(Array.fill(7)(1.0f)))
          loaded.close()
          assertTrue(pred.isSuccess)
        case _ => assertTrue(false)
    },
    test("predict with single sample returns array of 1") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val result = model.predict(Array(Array.fill(7)(0.5f)))
      assertTrue(result.isSuccess && result.get.length == 1)
    },
    test("classification: fit with class indices then predict") {
      val arch = dsl.Sequential(4)(dsl.Dense(8, dsl.ReLU), dsl.Output(3, dsl.MAE, dsl.Softmax))
        .withOptimizer(OptimizerDef.SGD(1e-4)).build
      val model = ZModel.create(arch)
      val feats = Array.fill(6)(Array.fill(4)(0.5f))
      val classIndices = Array(0, 0, 1, 1, 2, 2)
      val fitResult = model.fit(feats, classIndices, 3, 0.001f)
      val pred = model.predict(feats)
      assertTrue(fitResult.isSuccess && pred.isSuccess && pred.get.length == 18)
    }
  )
