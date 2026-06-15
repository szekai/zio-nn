package zio.nn.evaluation

/**
 * Pure evaluation utilities for classification and regression algorithms.
 *
 * Classification metrics operate on zero-based integer class labels. For ROC,
 * precision-recall, log loss, and Brier score, labels are expected to be binary
 * values in {0, 1} and scores represent the positive-class probability.
 */
object EvaluationMetrics {

  private final case class ClassMetrics(
    precision: Double,
    recall: Double,
    f1: Double,
    support: Int
  )

  private final case class ConfusionStats(
    rowSums: Array[Int],
    colSums: Array[Int],
    total: Int,
    correct: Int
  )

  /**
   * Build a confusion matrix where rows are actual labels and columns are predicted labels.
   */
  def confusionMatrix(
    predictions: Array[Int],
    labels: Array[Int],
    numClasses: Int
  ): Array[Array[Int]] = {
    require(numClasses > 0, s"numClasses must be positive, found $numClasses")
    validateLengths(predictions, labels, "predictions", "labels")

    val matrix = Array.ofDim[Int](numClasses, numClasses)
    var i = 0
    while (i < predictions.length) {
      val predicted = predictions(i)
      val actual = labels(i)

      validateClassIndex(actual, numClasses, "labels", i)
      validateClassIndex(predicted, numClasses, "predictions", i)

      matrix(actual)(predicted) += 1
      i += 1
    }

    matrix
  }

  /**
   * Returns macro-averaged precision, recall, F1, and overall accuracy.
   *
   * The tuple order matches EvaluationResult fields:
   * (precision, recall, f1, accuracy)
   */
  def calculateMetrics(confusionMatrix: Array[Array[Int]]): (Double, Double, Double, Double) = {
    if (confusionMatrix.isEmpty) {
      (0.0, 0.0, 0.0, 0.0)
    } else {
      validateConfusionMatrix(confusionMatrix)
      val stats = buildConfusionStats(confusionMatrix)
      val perClass = computeClassMetrics(confusionMatrix, stats)
      val macroMetrics = macroAverage(perClass)
      val accuracyValue = accuracy(stats)
      (macroMetrics._1, macroMetrics._2, macroMetrics._3, accuracyValue)
    }
  }

  /**
   * Logarithmic loss for binary probabilistic predictions.
   */
  def logLoss(predictions: Array[Double], labels: Array[Int]): Double = {
    validateProbabilityClassificationInputs(predictions, labels)

    if (predictions.isEmpty) {
      0.0
    } else {
      val epsilon = 1e-15
      var totalLoss = 0.0
      var i = 0

      while (i < predictions.length) {
        val probability = clipProbability(predictions(i), epsilon)
        if (labels(i) == 1) totalLoss -= math.log(probability)
        else totalLoss -= math.log(1.0 - probability)
        i += 1
      }

      totalLoss / predictions.length.toDouble
    }
  }

  /**
   * Brier score for binary probabilistic predictions.
   */
  def brierScore(predictions: Array[Double], labels: Array[Int]): Double = {
    validateProbabilityClassificationInputs(predictions, labels)

    if (predictions.isEmpty) {
      0.0
    } else {
      var squaredErrorSum = 0.0
      var i = 0

      while (i < predictions.length) {
        val error = predictions(i) - labels(i).toDouble
        squaredErrorSum += error * error
        i += 1
      }

      squaredErrorSum / predictions.length.toDouble
    }
  }

  /**
   * Explained variance regression score.
   */
  def explainedVariance(predictions: Array[Double], labels: Array[Double]): Double = {
    validateRegressionInputs(predictions, labels)

    if (predictions.isEmpty) {
      0.0
    } else {
      val labelMean = mean(labels)
      val count = predictions.length.toDouble
      var residualSum = 0.0
      var residualSquaredSum = 0.0
      var labelSquaredDeviationSum = 0.0
      var i = 0

      while (i < predictions.length) {
        val residual = labels(i) - predictions(i)
        val labelDeviation = labels(i) - labelMean

        residualSum += residual
        residualSquaredSum += residual * residual
        labelSquaredDeviationSum += labelDeviation * labelDeviation
        i += 1
      }

      val residualMean = residualSum / count
      val rawResidualVariance = residualSquaredSum / count - residualMean * residualMean
      val residualVariance = if (rawResidualVariance < 0.0) 0.0 else rawResidualVariance
      val labelVariance = labelSquaredDeviationSum / count

      if (labelVariance <= 0.0) {
        if (residualVariance <= 0.0) 1.0 else 0.0
      } else {
        1.0 - residualVariance / labelVariance
      }
    }
  }

  /**
   * Mean absolute error.
   */
  def mae(predictions: Array[Double], labels: Array[Double]): Double = {
    validateRegressionInputs(predictions, labels)

    if (predictions.isEmpty) {
      0.0
    } else {
      var absoluteErrorSum = 0.0
      var i = 0

      while (i < predictions.length) {
        absoluteErrorSum += math.abs(predictions(i) - labels(i))
        i += 1
      }

      absoluteErrorSum / predictions.length.toDouble
    }
  }

  /**
   * Root mean squared error.
   */
  def rmse(predictions: Array[Double], labels: Array[Double]): Double = {
    validateRegressionInputs(predictions, labels)

    if (predictions.isEmpty) {
      0.0
    } else {
      var squaredErrorSum = 0.0
      var i = 0

      while (i < predictions.length) {
        val error = predictions(i) - labels(i)
        squaredErrorSum += error * error
        i += 1
      }

      math.sqrt(squaredErrorSum / predictions.length.toDouble)
    }
  }

  /**
   * R² coefficient of determination.
   */
  def r2Score(predictions: Array[Double], labels: Array[Double]): Double = {
    validateRegressionInputs(predictions, labels)

    if (predictions.isEmpty) {
      0.0
    } else {
      val labelMean = mean(labels)
      var residualSquaredSum = 0.0
      var totalSquaredSum = 0.0
      var i = 0

      while (i < predictions.length) {
        val residual = labels(i) - predictions(i)
        val labelDeviation = labels(i) - labelMean

        residualSquaredSum += residual * residual
        totalSquaredSum += labelDeviation * labelDeviation
        i += 1
      }

      if (totalSquaredSum <= 0.0) {
        if (residualSquaredSum <= 0.0) 1.0 else 0.0
      } else {
        1.0 - residualSquaredSum / totalSquaredSum
      }
    }
  }

  /**
   * Per-class F1 scores keyed by class index.
   */
  def f1Score(predictions: Array[Int], labels: Array[Int]): Map[Int, Double] = {
    val summary = classificationMetricSummary(predictions, labels)
    val perClass = summary._1
    var scores = Map.empty[Int, Double]
    var classIndex = 0

    while (classIndex < perClass.length) {
      scores = scores.updated(classIndex, perClass(classIndex).f1)
      classIndex += 1
    }

    scores
  }

  /**
   * Macro-averaged F1 score.
   */
  def macroF1(predictions: Array[Int], labels: Array[Int]): Double = {
    val perClass = classificationMetricSummary(predictions, labels)._1
    macroAverage(perClass)._3
  }

  /**
   * Micro-averaged F1 score.
   */
  def microF1(predictions: Array[Int], labels: Array[Int]): Double = {
    val stats = classificationMetricSummary(predictions, labels)._2
    microAverage(stats)._3
  }

  /**
   * Support-weighted F1 score.
   */
  def weightedF1(predictions: Array[Int], labels: Array[Int]): Double = {
    val perClass = classificationMetricSummary(predictions, labels)._1
    var weightedSum = 0.0
    var totalSupport = 0
    var i = 0

    while (i < perClass.length) {
      weightedSum += perClass(i).f1 * perClass(i).support.toDouble
      totalSupport += perClass(i).support
      i += 1
    }

    safeDivide(weightedSum, totalSupport.toDouble)
  }

  /**
   * Compute ROC points as (false-positive-rate, true-positive-rate, threshold).
   */
  def rocCurve(
    scores: Array[Double],
    labels: Array[Int],
    thresholds: Array[Double]
  ): List[(Double, Double, Double)] = {
    validateBinaryInputs(scores, labels)

    val preparedThresholds =
      if (thresholds.isEmpty) uniqueSortedDescending(scores)
      else sortDescending(thresholds.clone())

    val (positivesCount, negativesCount) = countBinaryLabels(labels)
    val positives = positivesCount.toDouble
    val negatives = negativesCount.toDouble

    var points = List.empty[(Double, Double, Double)]
    var i = 0
    while (i < preparedThresholds.length) {
      val threshold = preparedThresholds(i)
      var tp = 0
      var fp = 0
      var j = 0

      while (j < scores.length) {
        if (scores(j) >= threshold) {
          if (labels(j) == 1) tp += 1 else fp += 1
        }
        j += 1
      }

      val tpr = if (positives == 0.0) 0.0 else tp.toDouble / positives
      val fpr = if (negatives == 0.0) 0.0 else fp.toDouble / negatives

      points = (fpr, tpr, threshold) :: points
      i += 1
    }

    points.reverse
  }

  /**
   * Trapezoidal area under curve.
   */
  def auc(fpr: Array[Double], tpr: Array[Double]): Double = {
    validateLengths(fpr, tpr, "fpr", "tpr")

    if (fpr.length < 2) {
      0.0
    } else {
      val sortedFpr = fpr.clone()
      val sortedTpr = tpr.clone()
      sortPairsAscending(sortedFpr, sortedTpr)

      var area = 0.0
      var i = 1
      while (i < sortedFpr.length) {
        val width = sortedFpr(i) - sortedFpr(i - 1)
        val height = sortedTpr(i) + sortedTpr(i - 1)
        area += width * height * 0.5
        i += 1
      }

      if (area < 0.0) 0.0 else area
    }
  }

  /**
   * Compute precision-recall curve points as (precision, recall).
   */
  def precisionRecallCurve(scores: Array[Double], labels: Array[Int]): List[(Double, Double)] = {
    validateBinaryInputs(scores, labels)

    val thresholds = uniqueSortedDescending(scores)
    val (positivesCount, _) = countBinaryLabels(labels)
    val positives = positivesCount.toDouble

    var points = List.empty[(Double, Double)]
    var i = 0
    while (i < thresholds.length) {
      val threshold = thresholds(i)
      var tp = 0
      var fp = 0
      var j = 0

      while (j < scores.length) {
        if (scores(j) >= threshold) {
          if (labels(j) == 1) tp += 1 else fp += 1
        }
        j += 1
      }

      val precision = safeDivide(tp.toDouble, tp.toDouble + fp.toDouble)
      val recall = if (positives == 0.0) 0.0 else tp.toDouble / positives

      points = (precision, recall) :: points
      i += 1
    }

    points.reverse
  }

  /**
   * Kolmogorov-Smirnov statistic for binary classification scores.
   */
  def ksStatistic(scores: Array[Double], labels: Array[Int]): Double = {
    binaryRocSummary(scores, labels)._2
  }

  /**
   * Gini coefficient derived from ROC AUC for binary classification scores.
   */
  def giniCoefficient(scores: Array[Double], labels: Array[Int]): Double = {
    val gini = 2.0 * binaryRocSummary(scores, labels)._1 - 1.0
    if (gini < -1.0) -1.0 else if (gini > 1.0) 1.0 else gini
  }

  /**
   * Cumulative lift curve points as (populationFraction, lift).
   */
  def liftCurve(scores: Array[Double], labels: Array[Int]): List[(Double, Double)] = {
    val prepared = prepareSortedBinaryRanking(scores, labels)
    val sortedLabels = prepared._2
    val positivesCount = prepared._3

    if (sortedLabels.isEmpty) {
      List.empty
    } else {
      val totalCount = sortedLabels.length.toDouble
      var cumulativePositives = 0
      var points = (0.0, 0.0) :: List.empty[(Double, Double)]
      var i = 0

      while (i < sortedLabels.length) {
        if (sortedLabels(i) == 1) cumulativePositives += 1

        val populationFraction = (i + 1).toDouble / totalCount
        val gain = if (positivesCount == 0) 0.0 else cumulativePositives.toDouble / positivesCount.toDouble
        val lift = if (populationFraction == 0.0) 0.0 else gain / populationFraction

        points = (populationFraction, lift) :: points
        i += 1
      }

      points.reverse
    }
  }

  /**
   * Cumulative gain curve points as (populationFraction, capturedPositiveFraction).
   */
  def gainCurve(scores: Array[Double], labels: Array[Int]): List[(Double, Double)] = {
    val prepared = prepareSortedBinaryRanking(scores, labels)
    val sortedLabels = prepared._2
    val positivesCount = prepared._3

    if (sortedLabels.isEmpty) {
      List.empty
    } else {
      val totalCount = sortedLabels.length.toDouble
      var cumulativePositives = 0
      var points = (0.0, 0.0) :: List.empty[(Double, Double)]
      var i = 0

      while (i < sortedLabels.length) {
        if (sortedLabels(i) == 1) cumulativePositives += 1

        val populationFraction = (i + 1).toDouble / totalCount
        val gain = if (positivesCount == 0) 0.0 else cumulativePositives.toDouble / positivesCount.toDouble

        points = (populationFraction, gain) :: points
        i += 1
      }

      points.reverse
    }
  }

  /**
   * Population Stability Index over 10 equal-width probability bins.
   */
  def psi(predictedProbs: Array[Double], actualProbs: Array[Double]): Double = {
    validateProbabilityArray(predictedProbs, "predictedProbs")
    validateProbabilityArray(actualProbs, "actualProbs")

    if (predictedProbs.isEmpty || actualProbs.isEmpty) {
      0.0
    } else {
      val numBins = 10
      val smoothing = 1e-10
      val predictedCounts = new Array[Int](numBins)
      val actualCounts = new Array[Int](numBins)

      var i = 0
      while (i < predictedProbs.length) {
        predictedCounts(probabilityBin(predictedProbs(i), numBins)) += 1
        i += 1
      }

      i = 0
      while (i < actualProbs.length) {
        actualCounts(probabilityBin(actualProbs(i), numBins)) += 1
        i += 1
      }

      var totalPsi = 0.0
      i = 0
      while (i < numBins) {
        val predictedShare = math.max(predictedCounts(i).toDouble / predictedProbs.length.toDouble, smoothing)
        val actualShare = math.max(actualCounts(i).toDouble / actualProbs.length.toDouble, smoothing)
        totalPsi += (actualShare - predictedShare) * math.log(actualShare / predictedShare)
        i += 1
      }

      totalPsi
    }
  }

  /**
   * Classification report with per-class metrics, macro/micro averages, accuracy,
   * and confusion matrix.
   */
  def classificationReport(predictions: Array[Int], labels: Array[Int]): String = {
    validateLengths(predictions, labels, "predictions", "labels")

    if (predictions.isEmpty) {
      "Classification Report\n=====================\nNo samples available.\n"
    } else {
      val numClasses = inferNumClasses(predictions, labels)
      val matrix = confusionMatrix(predictions, labels, numClasses)
      val stats = buildConfusionStats(matrix)
      val perClass = computeClassMetrics(matrix, stats)
      val macroMetrics = macroAverage(perClass)
      val micro = microAverage(stats)
      val accuracyValue = accuracy(stats)

      val builder = new StringBuilder(256 + numClasses * 64)
      builder.append("Classification Report\n")
      builder.append("label              precision      recall    f1-score      support\n")
      builder.append("----------------------------------------------------------------\n")

      var classIndex = 0
      while (classIndex < numClasses) {
        val metrics = perClass(classIndex)
        appendMetricRow(builder, classIndex.toString, metrics.precision, metrics.recall, metrics.f1, metrics.support)
        classIndex += 1
      }

      builder.append("----------------------------------------------------------------\n")
      appendAccuracyRow(builder, accuracyValue, stats.total)
      appendMetricRow(builder, "macro avg", macroMetrics._1, macroMetrics._2, macroMetrics._3, stats.total)
      appendMetricRow(builder, "micro avg", micro._1, micro._2, micro._3, stats.total)

      builder.append("\nConfusion Matrix (rows=actual, cols=predicted)\n")
      appendConfusionMatrix(builder, matrix)
      builder.toString
    }
  }

  private def computeClassMetrics(
    confusionMatrix: Array[Array[Int]],
    stats: ConfusionStats
  ): Array[ClassMetrics] = {
    val numClasses = confusionMatrix.length
    val metrics = new Array[ClassMetrics](numClasses)

    var classIndex = 0
    while (classIndex < numClasses) {
      val tp = confusionMatrix(classIndex)(classIndex).toDouble
      val fp = stats.colSums(classIndex).toDouble - tp
      val fn = stats.rowSums(classIndex).toDouble - tp

      val precision = safeDivide(tp, tp + fp)
      val recall = safeDivide(tp, tp + fn)
      val f1 = safeF1(precision, recall)

      metrics(classIndex) = ClassMetrics(
        precision = precision,
        recall = recall,
        f1 = f1,
        support = stats.rowSums(classIndex)
      )
      classIndex += 1
    }

    metrics
  }

  private def buildConfusionStats(confusionMatrix: Array[Array[Int]]): ConfusionStats = {
    val numClasses = confusionMatrix.length
    val rowSums = new Array[Int](numClasses)
    val colSums = new Array[Int](numClasses)

    var total = 0
    var correct = 0
    var row = 0
    while (row < numClasses) {
      var col = 0
      while (col < numClasses) {
        val value = confusionMatrix(row)(col)
        rowSums(row) += value
        colSums(col) += value
        total += value
        if (row == col) correct += value
        col += 1
      }
      row += 1
    }

    ConfusionStats(rowSums = rowSums, colSums = colSums, total = total, correct = correct)
  }

  private def macroAverage(metrics: Array[ClassMetrics]): (Double, Double, Double) = {
    if (metrics.isEmpty) {
      (0.0, 0.0, 0.0)
    } else {
      var precisionSum = 0.0
      var recallSum = 0.0
      var f1Sum = 0.0
      var i = 0
      while (i < metrics.length) {
        precisionSum += metrics(i).precision
        recallSum += metrics(i).recall
        f1Sum += metrics(i).f1
        i += 1
      }

      val count = metrics.length.toDouble
      (precisionSum / count, recallSum / count, f1Sum / count)
    }
  }

  private def microAverage(stats: ConfusionStats): (Double, Double, Double) = {
    val tp = stats.correct.toDouble
    val fp = (stats.total - stats.correct).toDouble
    val fn = (stats.total - stats.correct).toDouble

    val precision = safeDivide(tp, tp + fp)
    val recall = safeDivide(tp, tp + fn)
    val f1 = safeF1(precision, recall)

    (precision, recall, f1)
  }

  private def accuracy(stats: ConfusionStats): Double = {
    safeDivide(stats.correct.toDouble, stats.total.toDouble)
  }

  private def classificationMetricSummary(
    predictions: Array[Int],
    labels: Array[Int]
  ): (Array[ClassMetrics], ConfusionStats) = {
    validateLengths(predictions, labels, "predictions", "labels")

    if (predictions.isEmpty) {
      (
        new Array[ClassMetrics](0),
        ConfusionStats(rowSums = new Array[Int](0), colSums = new Array[Int](0), total = 0, correct = 0)
      )
    } else {
      val numClasses = inferNumClasses(predictions, labels)
      val matrix = confusionMatrix(predictions, labels, numClasses)
      val stats = buildConfusionStats(matrix)
      (computeClassMetrics(matrix, stats), stats)
    }
  }

  private def inferNumClasses(predictions: Array[Int], labels: Array[Int]): Int = {
    var maxClass = -1
    var i = 0
    while (i < labels.length) {
      val actual = labels(i)
      val predicted = predictions(i)

      require(actual >= 0, s"labels contain negative class index $actual at position $i")
      require(predicted >= 0, s"predictions contain negative class index $predicted at position $i")

      if (actual > maxClass) maxClass = actual
      if (predicted > maxClass) maxClass = predicted
      i += 1
    }

    maxClass + 1
  }

  private def countBinaryLabels(labels: Array[Int]): (Int, Int) = {
    var positives = 0
    var negatives = 0
    var i = 0
    while (i < labels.length) {
      if (labels(i) == 1) positives += 1 else negatives += 1
      i += 1
    }
    (positives, negatives)
  }

  private def validateBinaryInputs(scores: Array[Double], labels: Array[Int]): Unit = {
    validateLengths(scores, labels, "scores", "labels")

    var i = 0
    while (i < scores.length) {
      val score = scores(i)
      val label = labels(i)
      require(!score.isNaN && !score.isInfinite, s"scores must be finite, found $score at position $i")
      require(label == 0 || label == 1, s"binary labels must be 0 or 1, found $label at position $i")
      i += 1
    }
  }

  private def validateProbabilityClassificationInputs(predictions: Array[Double], labels: Array[Int]): Unit = {
    validateLengths(predictions, labels, "predictions", "labels")
    validateProbabilityArray(predictions, "predictions")
    validateBinaryLabels(labels, "labels")
  }

  private def validateRegressionInputs(predictions: Array[Double], labels: Array[Double]): Unit = {
    validateLengths(predictions, labels, "predictions", "labels")
    validateFiniteArray(predictions, "predictions")
    validateFiniteArray(labels, "labels")
  }

  private def validateBinaryLabels(labels: Array[Int], name: String): Unit = {
    var i = 0
    while (i < labels.length) {
      val label = labels(i)
      require(label == 0 || label == 1, s"$name must contain only 0 or 1, found $label at position $i")
      i += 1
    }
  }

  private def validateFiniteArray(values: Array[Double], name: String): Unit = {
    var i = 0
    while (i < values.length) {
      val value = values(i)
      require(!value.isNaN && !value.isInfinite, s"$name must be finite, found $value at position $i")
      i += 1
    }
  }

  private def prepareSortedBinaryRanking(scores: Array[Double], labels: Array[Int]): (Array[Double], Array[Int], Int) = {
    validateBinaryInputs(scores, labels)

    val sortedScores = scores.clone()
    val sortedLabels = labels.clone()
    sortScoresAndLabelsDescending(sortedScores, sortedLabels)
    val positivesCount = countBinaryLabels(sortedLabels)._1

    (sortedScores, sortedLabels, positivesCount)
  }

  private def binaryRocSummary(scores: Array[Double], labels: Array[Int]): (Double, Double) = {
    val prepared = prepareSortedBinaryRanking(scores, labels)
    val sortedScores = prepared._1
    val sortedLabels = prepared._2
    val positivesCount = prepared._3
    val negativesCount = sortedLabels.length - positivesCount

    if (sortedScores.isEmpty || positivesCount == 0 || negativesCount == 0) {
      (0.0, 0.0)
    } else {
      val positives = positivesCount.toDouble
      val negatives = negativesCount.toDouble

      var cumulativePositives = 0.0
      var cumulativeNegatives = 0.0
      var previousTpr = 0.0
      var previousFpr = 0.0
      var aucArea = 0.0
      var maxDifference = 0.0
      var i = 0

      while (i < sortedScores.length) {
        val currentScore = sortedScores(i)

        while (i < sortedScores.length && java.lang.Double.compare(sortedScores(i), currentScore) == 0) {
          if (sortedLabels(i) == 1) cumulativePositives += 1.0 else cumulativeNegatives += 1.0
          i += 1
        }

        val tpr = cumulativePositives / positives
        val fpr = cumulativeNegatives / negatives
        val difference = math.abs(tpr - fpr)

        aucArea += (fpr - previousFpr) * (tpr + previousTpr) * 0.5
        if (difference > maxDifference) maxDifference = difference

        previousTpr = tpr
        previousFpr = fpr
      }

      (if (aucArea < 0.0) 0.0 else aucArea, maxDifference)
    }
  }

  private def validateProbabilityArray(values: Array[Double], name: String): Unit = {
    var i = 0
    while (i < values.length) {
      val value = values(i)
      require(!value.isNaN && !value.isInfinite, s"$name must be finite, found $value at position $i")
      require(value >= 0.0 && value <= 1.0, s"$name must be in [0, 1], found $value at position $i")
      i += 1
    }
  }

  private def validateConfusionMatrix(confusionMatrix: Array[Array[Int]]): Unit = {
    val size = confusionMatrix.length
    var row = 0
    while (row < size) {
      require(
        confusionMatrix(row).length == size,
        s"confusionMatrix must be square: row $row has length ${confusionMatrix(row).length}, expected $size"
      )

      var col = 0
      while (col < size) {
        require(
          confusionMatrix(row)(col) >= 0,
          s"confusionMatrix cannot contain negative counts, found ${confusionMatrix(row)(col)} at ($row, $col)"
        )
        col += 1
      }
      row += 1
    }
  }

  private def validateClassIndex(value: Int, numClasses: Int, source: String, index: Int): Unit = {
    require(
      value >= 0 && value < numClasses,
      s"$source contain class index $value at position $index, expected range [0, ${numClasses - 1}]"
    )
  }

  private def validateLengths(left: Array[?], right: Array[?], leftName: String, rightName: String): Unit = {
    require(
      left.length == right.length,
      s"$leftName and $rightName must have the same length: ${left.length} != ${right.length}"
    )
  }

  private def uniqueSortedDescending(values: Array[Double]): Array[Double] = {
    if (values.isEmpty) {
      new Array[Double](0)
    } else {
      val copy = values.clone()
      java.util.Arrays.sort(copy)

      var uniqueCount = 1
      var i = 1
      while (i < copy.length) {
        if (java.lang.Double.compare(copy(i), copy(uniqueCount - 1)) != 0) {
          copy(uniqueCount) = copy(i)
          uniqueCount += 1
        }
        i += 1
      }

      val result = new Array[Double](uniqueCount)
      i = 0
      while (i < uniqueCount) {
        result(i) = copy(uniqueCount - 1 - i)
        i += 1
      }

      result
    }
  }

  private def sortDescending(values: Array[Double]): Array[Double] = {
    java.util.Arrays.sort(values)
    reverseInPlace(values)
    values
  }

  private def reverseInPlace(values: Array[Double]): Unit = {
    var left = 0
    var right = values.length - 1
    while (left < right) {
      val tmp = values(left)
      values(left) = values(right)
      values(right) = tmp
      left += 1
      right -= 1
    }
  }

  private def sortPairsAscending(xs: Array[Double], ys: Array[Double]): Unit = {
    var i = 1
    while (i < xs.length) {
      val x = xs(i)
      val y = ys(i)
      var j = i - 1
      while (j >= 0 && xs(j) > x) {
        xs(j + 1) = xs(j)
        ys(j + 1) = ys(j)
        j -= 1
      }
      xs(j + 1) = x
      ys(j + 1) = y
      i += 1
    }
  }

  private def sortScoresAndLabelsDescending(scores: Array[Double], labels: Array[Int]): Unit = {
    var i = 1
    while (i < scores.length) {
      val score = scores(i)
      val label = labels(i)
      var j = i - 1

      while (j >= 0 && scores(j) < score) {
        scores(j + 1) = scores(j)
        labels(j + 1) = labels(j)
        j -= 1
      }

      scores(j + 1) = score
      labels(j + 1) = label
      i += 1
    }
  }

  private def probabilityBin(value: Double, numBins: Int): Int = {
    val rawIndex = (value * numBins).toInt
    if (rawIndex < 0) 0 else if (rawIndex >= numBins) numBins - 1 else rawIndex
  }

  private def clipProbability(value: Double, epsilon: Double): Double = {
    if (value <= epsilon) epsilon
    else if (value >= 1.0 - epsilon) 1.0 - epsilon
    else value
  }

  private def mean(values: Array[Double]): Double = {
    if (values.isEmpty) {
      0.0
    } else {
      var sum = 0.0
      var i = 0
      while (i < values.length) {
        sum += values(i)
        i += 1
      }
      sum / values.length.toDouble
    }
  }

  private def safeDivide(numerator: Double, denominator: Double): Double = {
    if (denominator == 0.0) 0.0 else numerator / denominator
  }

  private def safeF1(precision: Double, recall: Double): Double = {
    val denominator = precision + recall
    if (denominator == 0.0) 0.0 else 2.0 * precision * recall / denominator
  }

  private def appendMetricRow(
    builder: StringBuilder,
    label: String,
    precision: Double,
    recall: Double,
    f1: Double,
    support: Int
  ): Unit = {
    builder.append(f"${label}%-14s ${precision}%12.4f ${recall}%11.4f ${f1}%11.4f ${support}%12d\n")
  }

  private def appendAccuracyRow(builder: StringBuilder, accuracy: Double, support: Int): Unit = {
    builder.append(f"${"accuracy"}%-14s ${""}%12s ${""}%11s ${accuracy}%11.4f ${support}%12d\n")
  }

  private def appendConfusionMatrix(builder: StringBuilder, confusionMatrix: Array[Array[Int]]): Unit = {
    builder.append(f"${"actual\\pred"}%-12s")

    var col = 0
    while (col < confusionMatrix.length) {
      builder.append(f"${col}%8d")
      col += 1
    }
    builder.append('\n')

    var row = 0
    while (row < confusionMatrix.length) {
      builder.append(f"${row}%-12d")
      col = 0
      while (col < confusionMatrix.length) {
        builder.append(f"${confusionMatrix(row)(col)}%8d")
        col += 1
      }
      builder.append('\n')
      row += 1
    }
  }
}

/**
 * Federated statistical utilities built from party-level aggregates and raw matrices.
 */
object FederatedStatistics {

  /**
   * Weighted global mean from party-level means and counts.
   */
  def federatedMean(partyMeans: Array[Double], partyCounts: Array[Int]): Double = {
    validateLengths(partyMeans, partyCounts, "partyMeans", "partyCounts")

    var weightedSum = 0.0
    var totalCount = 0L
    var i = 0
    while (i < partyMeans.length) {
      val mean = partyMeans(i)
      val count = partyCounts(i)

      require(!mean.isNaN && !mean.isInfinite, s"partyMeans must be finite, found $mean at position $i")
      require(count >= 0, s"partyCounts must be non-negative, found $count at position $i")

      weightedSum += mean * count.toDouble
      totalCount += count.toLong
      i += 1
    }

    if (totalCount == 0L) 0.0 else weightedSum / totalCount.toDouble
  }

  /**
   * Global population variance from party means, party variances, and counts.
   */
  def federatedVariance(
    partyMeans: Array[Double],
    partyVariances: Array[Double],
    partyCounts: Array[Int]
  ): Double = {
    validateLengths(partyMeans, partyVariances, "partyMeans", "partyVariances")
    validateLengths(partyMeans, partyCounts, "partyMeans", "partyCounts")

    val globalMean = federatedMean(partyMeans, partyCounts)
    var weightedVariance = 0.0
    var totalCount = 0L
    var i = 0

    while (i < partyMeans.length) {
      val mean = partyMeans(i)
      val variance = partyVariances(i)
      val count = partyCounts(i)

      require(!variance.isNaN && !variance.isInfinite, s"partyVariances must be finite, found $variance at position $i")
      require(variance >= 0.0, s"partyVariances must be non-negative, found $variance at position $i")

      val meanDelta = mean - globalMean
      weightedVariance += count.toDouble * (variance + meanDelta * meanDelta)
      totalCount += count.toLong
      i += 1
    }

    if (totalCount == 0L) 0.0 else weightedVariance / totalCount.toDouble
  }

  /**
   * Global population standard deviation from party means, variances, and counts.
   */
  def federatedStdDeviation(
    partyMeans: Array[Double],
    partyVariances: Array[Double],
    partyCounts: Array[Int]
  ): Double = {
    val variance = federatedVariance(partyMeans, partyVariances, partyCounts)
    if (variance <= 0.0) 0.0 else math.sqrt(variance)
  }

  /**
   * Pearson correlation matrix where rows are observations and columns are variables.
   */
  def correlationMatrix(data: Array[Array[Double]]): Array[Array[Double]] = {
    validateMatrix(data)

    if (data.isEmpty || data(0).isEmpty) {
      Array.ofDim[Double](0, 0)
    } else {
      val rowCount = data.length
      val columnCount = data(0).length
      val means = new Array[Double](columnCount)
      val variances = new Array[Double](columnCount)
      val correlations = Array.ofDim[Double](columnCount, columnCount)

      var col = 0
      while (col < columnCount) {
        var sum = 0.0
        var row = 0
        while (row < rowCount) {
          sum += data(row)(col)
          row += 1
        }
        means(col) = sum / rowCount.toDouble
        col += 1
      }

      col = 0
      while (col < columnCount) {
        var squaredDeviationSum = 0.0
        var row = 0
        while (row < rowCount) {
          val deviation = data(row)(col) - means(col)
          squaredDeviationSum += deviation * deviation
          row += 1
        }
        variances(col) = squaredDeviationSum / rowCount.toDouble
        col += 1
      }

      var i = 0
      while (i < columnCount) {
        correlations(i)(i) = 1.0
        var j = i + 1
        while (j < columnCount) {
          var covariance = 0.0
          var row = 0
          while (row < rowCount) {
            covariance += (data(row)(i) - means(i)) * (data(row)(j) - means(j))
            row += 1
          }
          covariance /= rowCount.toDouble

          val denominator = math.sqrt(variances(i) * variances(j))
          val correlation = if (denominator == 0.0) 0.0 else covariance / denominator

          correlations(i)(j) = correlation
          correlations(j)(i) = correlation
          j += 1
        }
        i += 1
      }

      correlations
    }
  }

  private def validateLengths(left: Array[?], right: Array[?], leftName: String, rightName: String): Unit = {
    require(
      left.length == right.length,
      s"$leftName and $rightName must have the same length: ${left.length} != ${right.length}"
    )
  }

  private def validateMatrix(data: Array[Array[Double]]): Unit = {
    if (data.isEmpty) {
      ()
    } else {
      require(data(0) != null, "data row 0 must not be null")
      val columnCount = data(0).length
      var row = 0
      while (row < data.length) {
        require(data(row) != null, s"data row $row must not be null")
        require(
          data(row).length == columnCount,
          s"data must be rectangular: row $row has length ${data(row).length}, expected $columnCount"
        )

        var col = 0
        while (col < columnCount) {
          val value = data(row)(col)
          require(!value.isNaN && !value.isInfinite, s"data must be finite, found $value at ($row, $col)")
          col += 1
        }
        row += 1
      }
    }
  }
}
