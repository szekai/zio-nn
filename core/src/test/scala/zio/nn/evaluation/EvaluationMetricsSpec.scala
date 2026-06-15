package zio.nn.evaluation

import zio.test.*
import zio.test.Assertion.*

object EvaluationMetricsSpec extends ZIOSpecDefault:

  def spec = suite("EvaluationMetrics")(
    // ── confusionMatrix ──
    suite("confusionMatrix")(
      test("2-class perfect classification") {
        val predictions = Array(0, 0, 1, 1)
        val labels      = Array(0, 0, 1, 1)
        val matrix      = EvaluationMetrics.confusionMatrix(predictions, labels, 2)
        assertTrue(
          matrix.length == 2,
          matrix(0)(0) == 2, matrix(0)(1) == 0,
          matrix(1)(0) == 0, matrix(1)(1) == 2
        )
      },
      test("2-class with errors") {
        val predictions = Array(0, 0, 1, 0)
        val labels      = Array(0, 1, 1, 0)
        val matrix      = EvaluationMetrics.confusionMatrix(predictions, labels, 2)
        assertTrue(
          matrix(0)(0) == 2, matrix(0)(1) == 0,
          matrix(1)(0) == 1, matrix(1)(1) == 1
        )
      },
      test("3-class all same") {
        val predictions = Array(2, 2, 2)
        val labels      = Array(2, 2, 2)
        val matrix      = EvaluationMetrics.confusionMatrix(predictions, labels, 3)
        assertTrue(
          matrix(2)(2) == 3,
          matrix(0)(0) == 0, matrix(1)(1) == 0
        )
      },
      test("empty input returns zero matrix") {
        val matrix = EvaluationMetrics.confusionMatrix(Array.empty[Int], Array.empty[Int], 3)
        assertTrue(
          matrix.length == 3,
          matrix(0)(0) == 0,
          matrix(2)(2) == 0
        )
      }
    ),

    // ── calculateMetrics ──
    suite("calculateMetrics")(
      test("perfect 2-class yields 1.0 precision, recall, f1, accuracy") {
        val matrix = Array(Array(5, 0), Array(0, 5))
        val (p, r, f, a) = EvaluationMetrics.calculateMetrics(matrix)
        assertTrue(
          math.abs(p - 1.0) < 1e-10,
          math.abs(r - 1.0) < 1e-10,
          math.abs(f - 1.0) < 1e-10,
          math.abs(a - 1.0) < 1e-10
        )
      },
      test("all wrong 2-class yields 0.0") {
        val matrix = Array(Array(0, 5), Array(5, 0))
        val (p, r, f, a) = EvaluationMetrics.calculateMetrics(matrix)
        assertTrue(
          math.abs(p) < 1e-10,
          math.abs(r) < 1e-10,
          math.abs(f) < 1e-10,
          math.abs(a) < 1e-10
        )
      },
      test("empty matrix returns zeros") {
        val (p, r, f, a) = EvaluationMetrics.calculateMetrics(Array.empty[Array[Int]])
        assertTrue(p == 0.0, r == 0.0, f == 0.0, a == 0.0)
      }
    ),

    // ── logLoss ──
    suite("logLoss")(
      test("perfect predictions yield 0") {
        val loss = EvaluationMetrics.logLoss(Array(1.0, 0.0, 1.0), Array(1, 0, 1))
        assertTrue(math.abs(loss) < 1e-10)
      },
      test("imperfect predictions yield positive loss") {
        val loss = EvaluationMetrics.logLoss(Array(0.9, 0.2), Array(1, 0))
        assertTrue(loss > 0.0)
      },
      test("empty input yields 0") {
        assertTrue(EvaluationMetrics.logLoss(Array.empty[Double], Array.empty[Int]) == 0.0)
      }
    ),

    // ── brierScore ──
    suite("brierScore")(
      test("perfect predictions yield 0") {
        val brier = EvaluationMetrics.brierScore(Array(1.0, 0.0), Array(1, 0))
        assertTrue(math.abs(brier) < 1e-10)
      },
      test("0.5 predictions yield 0.25") {
        val brier = EvaluationMetrics.brierScore(Array(0.5, 0.5), Array(1, 0))
        assertTrue(math.abs(brier - 0.25) < 1e-10)
      }
    ),

    // ── Regression metrics ──
    suite("regression")(
      test("explainedVariance perfect") {
        val ev = EvaluationMetrics.explainedVariance(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0))
        assertTrue(math.abs(ev - 1.0) < 1e-10)
      },
      test("explainedVariance empty returns 0") {
        assertTrue(EvaluationMetrics.explainedVariance(Array.empty[Double], Array.empty[Double]) == 0.0)
      },
      test("MAE perfect") {
        assertTrue(EvaluationMetrics.mae(Array(1.0, 2.0), Array(1.0, 2.0)) == 0.0)
      },
      test("MAE with error") {
        assertTrue(math.abs(EvaluationMetrics.mae(Array(1.0, 2.0), Array(3.0, 4.0)) - 2.0) < 1e-10)
      },
      test("RMSE perfect") {
        assertTrue(EvaluationMetrics.rmse(Array(1.0, 2.0), Array(1.0, 2.0)) == 0.0)
      },
      test("RMSE with uniform error") {
        assertTrue(math.abs(EvaluationMetrics.rmse(Array(0.0, 0.0), Array(1.0, 1.0)) - 1.0) < 1e-10)
      },
      test("R² perfect") {
        val r2 = EvaluationMetrics.r2Score(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0))
        assertTrue(math.abs(r2 - 1.0) < 1e-10)
      },
      test("R² mean predictor yields 0") {
        val r2 = EvaluationMetrics.r2Score(Array(2.0, 2.0, 2.0), Array(1.0, 2.0, 3.0))
        assertTrue(math.abs(r2) < 1e-10)
      }
    ),

    // ── F1 variants ──
    suite("F1 variants")(
      test("f1Score per-class perfect") {
        val scores = EvaluationMetrics.f1Score(Array(0, 0, 1, 1), Array(0, 0, 1, 1))
        assertTrue(
          math.abs(scores(0) - 1.0) < 1e-10,
          math.abs(scores(1) - 1.0) < 1e-10
        )
      },
      test("macroF1 perfect") {
        val f1 = EvaluationMetrics.macroF1(Array(0, 0, 1, 1), Array(0, 0, 1, 1))
        assertTrue(math.abs(f1 - 1.0) < 1e-10)
      },
      test("microF1 perfect") {
        val f1 = EvaluationMetrics.microF1(Array(0, 0, 1, 1), Array(0, 0, 1, 1))
        assertTrue(math.abs(f1 - 1.0) < 1e-10)
      },
      test("weightedF1 perfect") {
        val f1 = EvaluationMetrics.weightedF1(Array(0, 0, 1, 1), Array(0, 0, 1, 1))
        assertTrue(math.abs(f1 - 1.0) < 1e-10)
      }
    ),

    // ── ROC / AUC ──
    suite("ROC / AUC")(
      test("AUC perfect separation") {
        val roc = EvaluationMetrics.rocCurve(
          Array(0.9, 0.8, 0.2, 0.1), Array(1, 1, 0, 0),
          Array(0.0, 0.25, 0.5, 0.75, 1.0)
        )
        assertTrue(roc.nonEmpty)
      },
      test("AUC trapezoidal area") {
        val area = EvaluationMetrics.auc(Array(0.0, 0.5, 1.0), Array(0.0, 1.0, 1.0))
        assertTrue(area > 0.0)
      },
      test("AUC empty input returns 0") {
        assertTrue(EvaluationMetrics.auc(Array.empty[Double], Array.empty[Double]) == 0.0)
      },
      test("AUC single point returns 0") {
        assertTrue(EvaluationMetrics.auc(Array(0.5), Array(0.5)) == 0.0)
      }
    ),

    // ── PR curve ──
    suite("precisionRecallCurve")(
      test("PR curve produces points") {
        val pr = EvaluationMetrics.precisionRecallCurve(Array(0.9, 0.8, 0.2, 0.1), Array(1, 1, 0, 0))
        assertTrue(pr.nonEmpty, pr.head._1 >= 0.0, pr.head._2 >= 0.0)
      }
    ),

    // ── KS / Gini ──
    suite("KS / Gini")(
      test("KS statistic is non-negative") {
        val ks = EvaluationMetrics.ksStatistic(Array(0.9, 0.8, 0.2, 0.1), Array(1, 1, 0, 0))
        assertTrue(ks >= 0.0, ks <= 1.0)
      },
      test("Gini coefficient is within [-1, 1]") {
        val gini = EvaluationMetrics.giniCoefficient(Array(0.9, 0.8, 0.2, 0.1), Array(1, 1, 0, 0))
        assertTrue(gini >= -1.0, gini <= 1.0)
      }
    ),

    // ── Lift / Gain ──
    suite("liftCurve / gainCurve")(
      test("liftCurve produces points") {
        val lift = EvaluationMetrics.liftCurve(Array(0.9, 0.8, 0.2, 0.1), Array(1, 1, 0, 0))
        assertTrue(lift.nonEmpty)
      },
      test("gainCurve produces points with head at 0") {
        val gain = EvaluationMetrics.gainCurve(Array(0.9, 0.8, 0.2, 0.1), Array(1, 1, 0, 0))
        assertTrue(gain.nonEmpty, gain.head._1 == 0.0, gain.head._2 == 0.0)
      }
    ),

    // ── PSI ──
    suite("PSI")(
      test("identical distributions yield 0") {
        val psi = EvaluationMetrics.psi(
          Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
          Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        )
        assertTrue(math.abs(psi) < 1e-10)
      },
      test("shifted distributions yield positive PSI") {
        val psi = EvaluationMetrics.psi(
          Array(0.1, 0.1, 0.1),
          Array(0.9, 0.9, 0.9)
        )
        assertTrue(psi > 0.0)
      },
      test("empty input yields 0") {
        assertTrue(
          EvaluationMetrics.psi(Array.empty[Double], Array.empty[Double]) == 0.0
        )
      }
    ),

    // ── classificationReport ──
    suite("classificationReport")(
      test("report includes key sections") {
        val report = EvaluationMetrics.classificationReport(Array(0, 0, 1, 1), Array(0, 0, 1, 1))
        assertTrue(
          report.contains("Classification Report"),
          report.contains("Confusion Matrix"),
          report.contains("accuracy"),
          report.contains("macro avg"),
          report.contains("micro avg")
        )
      },
      test("empty input returns fallback message") {
        val report = EvaluationMetrics.classificationReport(Array.empty[Int], Array.empty[Int])
        assertTrue(report.contains("No samples available"))
      }
    ),

    // ── FederatedStatistics ──
    suite("FederatedStatistics")(
      test("federatedMean simple average") {
        val mean = FederatedStatistics.federatedMean(Array(1.0, 3.0), Array(2, 2))
        assertTrue(math.abs(mean - 2.0) < 1e-10)
      },
      test("federatedMean weighted by count") {
        val mean = FederatedStatistics.federatedMean(Array(0.0, 4.0), Array(3, 1))
        assertTrue(math.abs(mean - 1.0) < 1e-10)
      },
      test("federatedMean empty returns 0") {
        assertTrue(FederatedStatistics.federatedMean(Array.empty[Double], Array.empty[Int]) == 0.0)
      },
      test("federatedVariance single party equals party variance") {
        val variance = FederatedStatistics.federatedVariance(Array(2.0), Array(1.0), Array(5))
        assertTrue(math.abs(variance - 1.0) < 1e-10)
      },
      test("federatedStdDeviation roundtrip") {
        val std = FederatedStatistics.federatedStdDeviation(Array(2.0), Array(4.0), Array(5))
        assertTrue(math.abs(std - 2.0) < 1e-10)
      },
      test("correlationMatrix 2x2 identity") {
        val corr = FederatedStatistics.correlationMatrix(
          Array(Array(1.0, 5.0), Array(2.0, 6.0), Array(3.0, 7.0))
        )
        assertTrue(
          corr.length == 2,
          math.abs(corr(0)(0) - 1.0) < 1e-10,
          math.abs(corr(1)(1) - 1.0) < 1e-10
        )
      },
      test("correlationMatrix empty input") {
        val corr = FederatedStatistics.correlationMatrix(Array.empty[Array[Double]])
        assertTrue(corr.length == 0)
      }
    ),

    // ── Edge cases ──
    suite("edge cases")(
      test("single class classification") {
        val predictions = Array(0, 0, 0)
        val labels      = Array(0, 0, 0)
        val matrix      = EvaluationMetrics.confusionMatrix(predictions, labels, 1)
        val (p, r, f, a) = EvaluationMetrics.calculateMetrics(matrix)
        assertTrue(
          matrix(0)(0) == 3,
          math.abs(p - 1.0) < 1e-10,
          math.abs(r - 1.0) < 1e-10,
          math.abs(f - 1.0) < 1e-10,
          math.abs(a - 1.0) < 1e-10
        )
      },
      test("all same prediction in f1Score") {
        val scores = EvaluationMetrics.f1Score(Array(0, 0, 0), Array(0, 0, 0))
        assertTrue(math.abs(scores(0) - 1.0) < 1e-10)
      }
    )
  )
