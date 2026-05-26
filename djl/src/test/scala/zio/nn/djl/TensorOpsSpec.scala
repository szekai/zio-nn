package zio.nn.djl

import zio.nn.TensorOps
import zio.test.*
import zio.test.Assertion.*

object TensorOpsSpec extends ZIOSpecDefault:

  private def approxEqual(left: Array[Double], right: Array[Double], tolerance: Double = 1e-5): Boolean =
    left.length == right.length && left.zip(right).forall { case (l, r) => math.abs(l - r) <= tolerance }

  def spec = suite("TensorOps (DJL)")(
    test("constructors and unary ops preserve backend parity") {
      for
        zeros <- TensorOps.zeros(Array(2L, 2L))
        ones <- TensorOps.ones(Array(2L, 2L))
        filled <- TensorOps.fill(Array(2L), 3.5)
        identity <- TensorOps.eye(3)
        base <- TensorOps.createDouble1D(Array(-1.0, 0.0, 4.0))
        abs <- TensorOps.abs(base)
        square <- TensorOps.square(base)
        sign <- TensorOps.sign(base)
        positive <- TensorOps.createDouble1D(Array(1.0, math.E))
        logged <- TensorOps.log(positive)
        sigmoid <- TensorOps.sigmoid(base)
        summed <- TensorOps.sum(base)
        mean <- TensorOps.mean(base)
        std <- TensorOps.std(base)
        zerosCount <- TensorOps.countZeros(base)
        norm <- TensorOps.norm(base)
        zerosArr <- TensorOps.toDoubleArray(zeros)
        onesArr <- TensorOps.toDoubleArray(ones)
        filledArr <- TensorOps.toDoubleArray(filled)
        identityArr <- TensorOps.toDoubleArray(identity)
        absArr <- TensorOps.toDoubleArray(abs)
        squareArr <- TensorOps.toDoubleArray(square)
        signArr <- TensorOps.toDoubleArray(sign)
        loggedArr <- TensorOps.toDoubleArray(logged)
        sigmoidArr <- TensorOps.toDoubleArray(sigmoid)
        summedArr <- TensorOps.toDoubleArray(summed)
        meanArr <- TensorOps.toDoubleArray(mean)
        stdArr <- TensorOps.toDoubleArray(std)
      yield assertTrue(
        approxEqual(zerosArr, Array(0.0, 0.0, 0.0, 0.0)),
        approxEqual(onesArr, Array(1.0, 1.0, 1.0, 1.0)),
        approxEqual(filledArr, Array(3.5, 3.5)),
        approxEqual(identityArr, Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
        approxEqual(absArr, Array(1.0, 0.0, 4.0)),
        approxEqual(squareArr, Array(1.0, 0.0, 16.0)),
        approxEqual(signArr, Array(-1.0, 0.0, 1.0)),
        approxEqual(loggedArr, Array(0.0, 1.0)),
        approxEqual(sigmoidArr, Array(1.0 / (1.0 + math.exp(1.0)), 0.5, 1.0 / (1.0 + math.exp(-4.0)))),
        approxEqual(summedArr, Array(3.0)),
        approxEqual(meanArr, Array(1.0)),
        approxEqual(stdArr, Array(math.sqrt(14.0 / 3.0))),
        zerosCount == 1,
        math.abs(norm - math.sqrt(17.0)) <= 1e-5
      )
    },
    test("indexing and shape helpers work for approved tensor ops") {
      for
        matrix <- TensorOps.createDouble(Array(Array(1.0, 2.0), Array(3.0, 4.0)))
        other <- TensorOps.createDouble(Array(Array(5.0, 6.0), Array(7.0, 8.0)))
        rhs <- TensorOps.createDouble1D(Array(5.0, 11.0))
        diagonal <- TensorOps.diagonal(matrix)
        element <- TensorOps.get(matrix, Array(1L, 0L))
        sliced <- TensorOps.slice(matrix, Array(0L, 0L), Array(2L, 1L))
        concatenated <- TensorOps.concatenate(matrix, other, 1)
        maxed <- TensorOps.maximum(matrix, other)
        solution <- TensorOps.solve(matrix, rhs)
        copied <- TensorOps.copy(matrix)
        vector <- TensorOps.createDouble1D(Array(1.0, 2.0, 2.0, 3.0))
        threshold <- TensorOps.createDouble1D(Array(1.0, 1.5, 2.0, 4.0))
        leftDot <- TensorOps.createDouble1D(Array(1.0, 2.0, 3.0))
        rightDot <- TensorOps.createDouble1D(Array(4.0, 5.0, 6.0))
        gathered <- TensorOps.gather(vector, Array(3, 0, 1))
        unique <- TensorOps.unique(vector)
        mask <- TensorOps.createDouble1D(Array(1.0, 0.0, 1.0, 0.0))
        filtered <- TensorOps.where(vector, mask)
        invertedMask <- TensorOps.not(mask)
        lte <- TensorOps.lessThanOrEqual(vector, threshold)
        dot <- TensorOps.dot(leftDot, rightDot)
        diagArr <- TensorOps.toDoubleArray(diagonal)
        slicedArr <- TensorOps.toDoubleArray(sliced)
        concatArr <- TensorOps.toDoubleArray(concatenated)
        maxArr <- TensorOps.toDoubleArray(maxed)
        solutionArr <- TensorOps.toDoubleArray(solution)
        copiedArr <- TensorOps.toDoubleArray(copied)
        gatheredArr <- TensorOps.toDoubleArray(gathered)
        uniqueArr <- TensorOps.toDoubleArray(unique)
        filteredArr <- TensorOps.toDoubleArray(filtered)
        invertedArr <- TensorOps.toDoubleArray(invertedMask)
        lteArr <- TensorOps.toDoubleArray(lte)
        dotArr <- TensorOps.toDoubleArray(dot)
      yield assertTrue(
        approxEqual(diagArr, Array(1.0, 4.0)),
        element == 3.0,
        approxEqual(slicedArr, Array(1.0, 3.0)),
        approxEqual(concatArr, Array(1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0)),
        approxEqual(maxArr, Array(5.0, 6.0, 7.0, 8.0)),
        approxEqual(solutionArr, Array(1.0, 2.0)),
        approxEqual(copiedArr, Array(1.0, 2.0, 3.0, 4.0)),
        approxEqual(gatheredArr, Array(3.0, 1.0, 2.0)),
        approxEqual(uniqueArr, Array(1.0, 2.0, 3.0)),
        approxEqual(filteredArr, Array(1.0, 2.0)),
        approxEqual(invertedArr, Array(0.0, 1.0, 0.0, 1.0)),
        approxEqual(lteArr, Array(1.0, 0.0, 1.0, 1.0)),
        approxEqual(dotArr, Array(32.0))
      )
    },
    test("shape and domain errors surface as failures") {
      val program = for
        vector <- TensorOps.createDouble1D(Array(1.0, 2.0, 3.0))
        nonPositive <- TensorOps.createDouble1D(Array(1.0, 0.0, -2.0))
        matrix <- TensorOps.createDouble(Array(Array(1.0, 2.0), Array(3.0, 4.0)))
        mismatch <- TensorOps.createDouble1D(Array(1.0, 0.0))
        badConcat <- TensorOps.createDouble(Array(Array(1.0), Array(2.0), Array(3.0)))
        concatFailure <- TensorOps.concatenate(matrix, badConcat, 1).exit
        logFailure <- TensorOps.log(nonPositive).exit
        gatherFailure <- TensorOps.gather(vector, Array(0, 4)).exit
        solveFailure <- TensorOps.solve(vector, mismatch).exit
        whereFailure <- TensorOps.where(vector, mismatch).exit
      yield assert(concatFailure)(fails(isSubtype[IllegalArgumentException](anything))) &&
        assert(logFailure)(fails(isSubtype[IllegalArgumentException](anything))) &&
        assert(gatherFailure)(fails(isSubtype[IllegalArgumentException](anything))) &&
        assert(solveFailure)(fails(isSubtype[IllegalArgumentException](anything))) &&
        assert(whereFailure)(fails(isSubtype[IllegalArgumentException](anything)))

      program
    }
  )
