package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object OnnxExportSpec extends ZIOSpecDefault:

  def spec = suite("ONNX Export (DL4J)")(
    test("toOnnx returns non-empty bytes for simple Dense model") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val result = Dl4jToOnnx.toOnnx(model.underlying)
      assertTrue(result.isSuccess && result.get.length > 0)
    },

    test("saveToFile writes valid ONNX file to disk") {
      val arch = dsl.Sequential(7)(dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val tmp = java.io.File.createTempFile("zionn", ".onnx")
      tmp.deleteOnExit()
      val result = Dl4jToOnnx.saveToFile(model.underlying, tmp.toPath)
      assertTrue(result.isSuccess && tmp.length() > 0)
    },

    test("toOnnx succeeds for LSTM model") {
      val arch = dsl.Sequential(7)(dsl.LSTM(64, dsl.Tanh), dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val result = Dl4jToOnnx.toOnnx(model.underlying)
      assertTrue(result.isSuccess && result.get.length > 0)
    },

    test("toOnnx succeeds for Dropout model (dropout is identity)") {
      val arch = dsl.Sequential(7)(dsl.Dropout(0.5), dsl.Dense(5), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val result = Dl4jToOnnx.toOnnx(model.underlying)
      assertTrue(result.isSuccess && result.get.length > 0)
    },

    test("toOnnx succeeds for Embedding model") {
      val arch = dsl.Sequential(1)(dsl.Embedding(1000, 128), dsl.Output(1, dsl.MAE)).build
      val model = ZModel.create(arch)
      val result = Dl4jToOnnx.toOnnx(model.underlying)
      assertTrue(result.isSuccess && result.get.length > 0)
    },

    test("ZModel.toOnnx companion method works") {
      val arch = dsl.Sequential(4)(dsl.Dense(3), dsl.Output(1)).build
      val model = ZModel.create(arch)
      val tmp = java.io.File.createTempFile("zionn", ".onnx")
      tmp.deleteOnExit()
      val result = ZModel.toOnnx(model.underlying, tmp.toPath)
      assertTrue(result.isSuccess && tmp.length() > 0)
    },

    test("toOnnx succeeds for BatchNormalization model") {
      val arch = dsl.Sequential(7)(
        dsl.Dense(32, dsl.ReLU), dsl.BatchNorm, dsl.Dense(16, dsl.ReLU), dsl.Output(1)
      ).build
      val model = ZModel.create(arch)
      val result = Dl4jToOnnx.toOnnx(model.underlying)
      assertTrue(result.isSuccess && result.get.length > 0)
    }
  )
