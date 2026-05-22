package zio.nn.dl4j

import zio.nn.*
import zio.nn.dsl.*
import zio.test.*
import zio.test.Assertion.*
import zio.*
import zio.nn.dl4j.zioApi.*

object ZIOApiSpec extends ZIOSpecDefault:

  def spec = suite("ZIO API (DL4J)")(
    test("create returns Scoped ZModel") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      for
        model <- zioApi.create(arch)
        _     <- ZIO.attemptBlocking(model.close()).orDie
      yield assertTrue(true)
    },
    test("predictZ works") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          result <- model.predictZ(Array.fill(2)(Array.fill(7)(0.5f)))
          _      <- ZIO.attemptBlocking(model.close()).orDie
        yield assertTrue(result.length == 2)
      }
    },
    test("predictDoubleZ works with Double precision") {
      val arch = Sequential(7)(Dense(5), Output(1)).build
      zioApi.create(arch).flatMap { model =>
        for
          result <- model.predictDoubleZ(Array.fill(1)(Array.fill(7)(0.5)))
          _      <- ZIO.attemptBlocking(model.close()).orDie
        yield assertTrue(result.length == 1)
      }
    }
  ).provideLayer(Scope.default)
