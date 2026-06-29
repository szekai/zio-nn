import sbt.{ given, * }
import java.net.URI

val scala3 = "3.8.3"
val zioV   = "2.1.25"
val djlV   = "0.36.0"
val dl4jV  = "1.0.0-M2.1"

scalaVersion := scala3
organization := "io.github.szekai"

homepage     := Some(URI("https://github.com/szekai/zio-nn"))
licenses     := Seq("Apache-2.0" -> URI("https://www.apache.org/licenses/LICENSE-2.0"))
developers   := List(
  Developer("szekai", "Sze Kai", "szekai@users.noreply.github.com", url("https://github.com/szekai"))
)
scmInfo      := Some(ScmInfo(
  url("https://github.com/szekai/zio-nn"),
  "scm:git:https://github.com/szekai/zio-nn.git",
  "scm:git:git@github.com:szekai/zio-nn.git"
))

// publishTo — managed by sbt-ci-release

val storchV  = "0.7.6-1.5.12"          // mullerhai fork (published on Maven Central)
val torchV   = "2.7.1-1.5.12"         // bundled PyTorch version in mullerhai fork
val javacppV = "1.5.12"               // JavaCPP version used by mullerhai fork

// ── Core: framework-agnostic architecture DSL ──────────
lazy val core = project
  .in(file("core"))
  .settings(
    name := "zio-nn-core",
    libraryDependencies ++= Seq(
      "dev.zio" %% "zio"                    % zioV,
      "dev.zio" %% "zio-streams"            % zioV,
      "dev.zio" %% "zio-config"             % "4.0.4",
      "dev.zio" %% "zio-config-typesafe"    % "4.0.4",
      "dev.zio" %% "zio-config-magnolia"    % "4.0.4",
      "dev.zio" %% "zio-json"              % "0.7.3",
      "dev.zio" %% "zio-test"               % zioV % Test,
      "dev.zio" %% "zio-test-sbt"           % zioV % Test
    )
  )

// ── DJL backend: AWS Deep Java Library ─────────────────
lazy val djl = project
  .in(file("djl"))
  .dependsOn(core)
  .settings(
    name := "zio-nn-djl",
    Test / fork := true,
    libraryDependencies ++= Seq(
      "dev.zio"           %% "zio"             % zioV,
      "dev.zio"           %% "zio-streams"     % zioV,
      "ai.djl"             % "api"             % djlV,
      "ai.djl.pytorch"     % "pytorch-engine"  % djlV,
      "ai.djl.huggingface" % "tokenizers"      % djlV,
      "dev.zio"           %% "zio-test"        % zioV % Test,
      "dev.zio"           %% "zio-test-sbt"    % zioV % Test
    )
  )

// ── DL4J backend: Eclipse Deeplearning4j ───────────────
lazy val dl4j = project
  .in(file("dl4j"))
  .dependsOn(core, vectordb % Test)
  .settings(
    name := "zio-nn-dl4j",
    libraryDependencies ++= Seq(
      "dev.zio"             %% "zio"                    % zioV,
      "dev.zio"             %% "zio-streams"            % zioV,
      "org.deeplearning4j"   % "deeplearning4j-core"    % dl4jV,
      "org.nd4j"             % "nd4j-native-platform"   % dl4jV,
      "dev.zio"             %% "zio-test"               % zioV % Test,
      "dev.zio"             %% "zio-test-sbt"           % zioV % Test
    )
  )

// ── DL4J Embeddings: Word2Vec training + pre-trained vector loading ──
lazy val embeddings = project
  .in(file("embeddings"))
  .dependsOn(core, dl4j)
  .settings(
    name := "zio-nn-dl4j-embeddings",
    libraryDependencies ++= Seq(
      "dev.zio"             %% "zio"                    % zioV,
      "dev.zio"             %% "zio-streams"            % zioV,
      "org.deeplearning4j"   % "deeplearning4j-nlp"     % dl4jV,
      "dev.zio"             %% "zio-test"               % zioV % Test,
      "dev.zio"             %% "zio-test-sbt"           % zioV % Test
    )
  )

// ── Vector Database: pgvector integration via Magnum ────
lazy val vectordb = project
  .in(file("vectordb"))
  .dependsOn(core)
  .settings(
    name := "zio-nn-vectordb",
    libraryDependencies ++= Seq(
      "dev.zio"            %% "zio"              % zioV,
      "dev.zio"            %% "zio-streams"      % zioV,
      "com.augustnagro"    %% "magnum"           % "1.3.0",
      "com.pgvector"        % "pgvector"         % "0.1.6",
      "org.postgresql"      % "postgresql"       % "42.7.11",
      "dev.zio"            %% "zio-test"         % zioV % Test,
      "dev.zio"            %% "zio-test-sbt"     % zioV % Test
    )
  )

// ── RAG: Retrieval-Augmented Generation pipeline ─────────
lazy val rag = project
  .in(file("rag"))
  .dependsOn(core, vectordb)
  .settings(
    name := "zio-nn-rag",
    libraryDependencies ++= Seq(
      "dev.zio"            %% "zio"                    % zioV,
      "dev.zio"            %% "zio-streams"            % zioV,
      "dev.zio"            %% "zio-json"               % "0.7.3",
      "dev.zio"            %% "zio-test"               % zioV % Test,
      "dev.zio"            %% "zio-test-sbt"           % zioV % Test
    )
  )

// ── Examples: end-to-end runnable apps ──────────────────
lazy val examples = project
  .in(file("examples"))
  .dependsOn(core, djl, dl4j, embeddings)
  .settings(
    name := "zio-nn-examples",
    publish / skip := true,
    libraryDependencies ++= Seq(
      "dev.zio" %% "zio"           % zioV,
      "dev.zio" %% "zio-streams"   % zioV
    )
  )

lazy val isMac: Boolean =
  sys.props.get("os.name").exists(_.toLowerCase.contains("mac"))

// ── Storch backend: PyTorch via JavaCPP ─────────────────
lazy val storch = project
  .in(file("storch"))
  .dependsOn(core)
  .settings(
    name := "zio-nn-storch",
    fork := true,
    Test / skip := !isMac,  // native libs only available on macOS
    libraryDependencies ++= Seq(
      "io.github.mullerhai" % "storch_core_3"   % storchV,
      "org.bytedeco"        % "javacpp"         % javacppV classifier "macosx-arm64",
      "org.bytedeco"        % "pytorch"         % torchV   classifier "macosx-arm64",
      "org.bytedeco"        % "openblas"        % "0.3.30-1.5.12" classifier "macosx-arm64",
      "dev.zio"            %% "zio"             % zioV,
      "dev.zio"            %% "zio-streams"     % zioV,
      "dev.zio"            %% "zio-test"        % zioV % Test,
      "dev.zio"            %% "zio-test-sbt"    % zioV % Test
    )
  )

// ── Root aggregate ─────────────────────────────────────
lazy val root = project
  .in(file("."))
  .aggregate(core, djl, dl4j, embeddings, vectordb, rag, examples, storch)
  .settings(
    name := "zio-nn",
    publish / skip := true
  )
