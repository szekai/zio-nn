import sbt.*

val scala3 = "3.8.3"
val zioV   = "2.1.25"
val djlV   = "0.36.0"
val dl4jV  = "1.0.0-M2.1"

ThisBuild / scalaVersion := scala3
ThisBuild / organization := "io.github.szekai"
ThisBuild / version      := "0.5.2"
ThisBuild / homepage     := Some(url("https://github.com/szekai/zio-nn"))
ThisBuild / licenses     := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / developers   := List(
  Developer("szekai", "Sze Kai", "szekai@users.noreply.github.com", url("https://github.com/szekai"))
)
ThisBuild / scmInfo      := Some(ScmInfo(
  url("https://github.com/szekai/zio-nn"),
  "scm:git:https://github.com/szekai/zio-nn.git",
  "scm:git:git@github.com:szekai/zio-nn.git"
))

// Sonatype / Maven Central
ThisBuild / sonatypeCredentialHost := "s01.oss.sonatype.org"
ThisBuild / sonatypeProfileName    := "io.github.szekai"
ThisBuild / publishTo              := sonatypePublishToBundle.value
ThisBuild / pomIncludeRepository   := { _ => false }

// Disable publishing for root aggregate
publish / skip := true

// ── Core: framework-agnostic architecture DSL ──────────
lazy val core = project
  .in(file("core"))
  .settings(
    name := "zio-nn-core",
    libraryDependencies ++= Seq(
      "dev.zio" %% "zio"     % zioV,
      "dev.zio" %% "zio-test" % zioV % Test,
      "dev.zio" %% "zio-test-sbt" % zioV % Test
    )
  )

// ── DJL backend: AWS Deep Java Library ─────────────────
lazy val djl = project
  .in(file("djl"))
  .dependsOn(core)
  .settings(
    name := "zio-nn-djl",
    libraryDependencies ++= Seq(
      "dev.zio"           %% "zio"             % zioV,
      "ai.djl"             % "api"             % djlV,
      "ai.djl.pytorch"     % "pytorch-engine"  % djlV,
      "dev.zio"           %% "zio-test"        % zioV % Test,
      "dev.zio"           %% "zio-test-sbt"    % zioV % Test
    )
  )

// ── DL4J backend: Eclipse Deeplearning4j ───────────────
lazy val dl4j = project
  .in(file("dl4j"))
  .dependsOn(core)
  .settings(
    name := "zio-nn-dl4j",
    libraryDependencies ++= Seq(
      "dev.zio"             %% "zio"                    % zioV,
      "org.deeplearning4j"   % "deeplearning4j-core"    % dl4jV,
      "org.nd4j"             % "nd4j-native-platform"   % dl4jV,
      "dev.zio"             %% "zio-test"               % zioV % Test,
      "dev.zio"             %% "zio-test-sbt"           % zioV % Test
    )
    )

// ── Root aggregate ─────────────────────────────────────
lazy val root = project
  .in(file("."))
  .aggregate(core, djl, dl4j)
  .settings(
    name := "zio-nn",
    publish / skip := true
  )
