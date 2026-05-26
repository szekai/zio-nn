package zio.nn.dl4j.tensor

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import zio.*

/** ZIO-wrapped tensor math operations for DL4J's INDArray.
  * Direct replacement for zenithfl's Nd4jZIO facade.
  *
  * Signatures match the DJL backend exactly — enabling zero-code-change
  * backend swap via `import zio.nn.TensorOps.*`.
  *
  * Usage (works identically with both DL4J and DJL backends):
  * {{{
  *   import zio.nn.TensorOps.*
  *   for
  *     a <- createDouble1D(Array(1.0, 2.0, 3.0))
  *     b <- createDouble1D(Array(0.5, 0.5, 0.5))
  *     c <- add(a, b)
  *     e <- toDoubleArray(c)
  *   yield e
  * }}}
  */
object TensorOps:

  private def fail[A](message: String): Task[A] =
    ZIO.fail(IllegalArgumentException(message))

  private def validateShape(targetShape: Array[Long], op: String): Task[Unit] =
    if targetShape.isEmpty then fail(s"$op requires at least one dimension")
    else if targetShape.exists(_ < 0) then fail(s"$op does not support negative dimensions: ${targetShape.mkString("[", ", ", "]")}")
    else if targetShape.length > 2 then fail(s"$op currently supports only 1D or 2D tensors, got rank ${targetShape.length}")
    else ZIO.unit

  private def createFromFlat(flat: Array[Double], targetShape: Array[Long], op: String): Task[INDArray] =
    for
      _ <- validateShape(targetShape, op)
      out <- targetShape.length match
        case 1 =>
          if flat.length != targetShape(0).toInt then fail(s"$op expected ${targetShape(0)} values but received ${flat.length}")
          else createDouble1D(flat)
        case 2 =>
          val rows = targetShape(0).toInt
          val cols = targetShape(1).toInt
          if flat.length != rows * cols then fail(s"$op expected ${rows * cols} values but received ${flat.length}")
          else createDouble(Array.tabulate(rows, cols)((r, c) => flat(r * cols + c)))
        case _ => fail(s"$op currently supports only 1D or 2D tensors")
    yield out

  private def normalizedBound(start: Long, end: Long, size: Long, axis: Int, op: String): Task[(Int, Int)] =
    val normalizedEnd = if end == -1 then size else end
    if start < 0 || normalizedEnd < start || normalizedEnd > size then
      fail(s"$op has invalid bounds on axis $axis: start=$start end=$end size=$size")
    else
      ZIO.succeed((start.toInt, normalizedEnd.toInt))

  private def requireSameShape(left: Array[Long], right: Array[Long], op: String): Task[Unit] =
    if left.sameElements(right) then ZIO.unit
    else fail(s"$op requires matching shapes: ${left.mkString("[", ", ", "]")} vs ${right.mkString("[", ", ", "]")}")

  private def solveLinearSystem(matrix: Array[Array[Double]], rhs: Array[Array[Double]]): Array[Array[Double]] =
    val n = matrix.length
    val rhsCols = rhs.headOption.map(_.length).getOrElse(0)
    val a = matrix.map(_.clone())
    val b = rhs.map(_.clone())

    for pivot <- 0 until n do
      val maxRow = (pivot until n).maxBy(row => math.abs(a(row)(pivot)))
      if math.abs(a(maxRow)(pivot)) < 1e-12 then
        throw IllegalArgumentException("solve requires a non-singular matrix")
      if maxRow != pivot then
        val tmpA = a(pivot)
        a(pivot) = a(maxRow)
        a(maxRow) = tmpA
        val tmpB = b(pivot)
        b(pivot) = b(maxRow)
        b(maxRow) = tmpB

      val pivotValue = a(pivot)(pivot)
      for col <- pivot until n do a(pivot)(col) /= pivotValue
      for col <- 0 until rhsCols do b(pivot)(col) /= pivotValue

      for row <- 0 until n if row != pivot do
        val factor = a(row)(pivot)
        for col <- pivot until n do a(row)(col) -= factor * a(pivot)(col)
        for col <- 0 until rhsCols do b(row)(col) -= factor * b(pivot)(col)

    b

  def zeros(targetShape: Array[Long]): Task[INDArray] =
    for
      _ <- validateShape(targetShape, "zeros")
      size = targetShape.product.toInt
      out <- createFromFlat(Array.fill(size)(0.0), targetShape, "zeros")
    yield out

  def ones(targetShape: Array[Long]): Task[INDArray] =
    for
      _ <- validateShape(targetShape, "ones")
      size = targetShape.product.toInt
      out <- createFromFlat(Array.fill(size)(1.0), targetShape, "ones")
    yield out

  def fill(targetShape: Array[Long], value: Double): Task[INDArray] =
    for
      _ <- validateShape(targetShape, "fill")
      size = targetShape.product.toInt
      out <- createFromFlat(Array.fill(size)(value), targetShape, "fill")
    yield out

  def eye(size: Long): Task[INDArray] =
    if size < 0 then fail(s"eye does not support negative size: $size")
    else
      val intSize = size.toInt
      createDouble(Array.tabulate(intSize, intSize)((r, c) => if r == c then 1.0 else 0.0))

  def create(data: Array[Array[Float]]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def create1D(data: Array[Float]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def createDouble(data: Array[Array[Double]]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def createDouble1D(data: Array[Double]): Task[INDArray] =
    ZIO.attemptBlocking(Nd4j.create(data))

  def add(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.add(b))

  def sub(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.sub(b))

  def mul(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.mul(b))

  def div(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.div(b))

  def matMul(a: INDArray, b: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.mmul(b))

  def dot(a: INDArray, b: INDArray): Task[INDArray] =
    for
      leftShape  <- shape(a)
      rightShape <- shape(b)
      out <-
        if leftShape.length == 1 && rightShape.length == 1 then
          if leftShape(0) != rightShape(0) then fail(s"dot requires vectors of same length: ${leftShape(0)} vs ${rightShape(0)}")
          else
            for
              left  <- toDoubleArray(a)
              right <- toDoubleArray(b)
              dotValue = left.zip(right).map(_ * _).sum
              out <- createDouble1D(Array(dotValue))
            yield out
        else if leftShape.length == 2 && rightShape.length == 2 then
          if leftShape(1) != rightShape(0) then fail(s"dot requires aligned matrix shapes: ${leftShape.mkString("[", ", ", "]")} vs ${rightShape.mkString("[", ", ", "]")}")
          else matMul(a, b)
        else fail(s"dot supports only 1D vectors or 2D matrices, got ${leftShape.length}D and ${rightShape.length}D")
    yield out

  def transpose(a: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.transpose())

  def sum(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      out    <- createDouble1D(Array(values.sum))
    yield out

  def mean(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      out <-
        if values.isEmpty then fail("mean requires at least one element")
        else createDouble1D(Array(values.sum / values.length))
    yield out

  def neg(a: INDArray): Task[INDArray] =
    ZIO.attemptBlocking(a.neg())

  def abs(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      targetShape <- shape(a)
      out <- createFromFlat(values.map(math.abs), targetShape, "abs")
    yield out

  def square(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      targetShape <- shape(a)
      out <- createFromFlat(values.map(v => v * v), targetShape, "square")
    yield out

  def sign(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      targetShape <- shape(a)
      out <- createFromFlat(values.map(v => math.signum(v)), targetShape, "sign")
    yield out

  def log(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      _ <-
        if values.forall(_ > 0.0) then ZIO.unit
        else fail("log requires all tensor elements to be > 0")
      targetShape <- shape(a)
      out <- createFromFlat(values.map(math.log), targetShape, "log")
    yield out

  def sigmoid(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      targetShape <- shape(a)
      out <- createFromFlat(values.map(v => 1.0 / (1.0 + math.exp(-v))), targetShape, "sigmoid")
    yield out

  def maximum(a: INDArray, b: INDArray): Task[INDArray] =
    for
      leftShape  <- shape(a)
      rightShape <- shape(b)
      _          <- requireSameShape(leftShape, rightShape, "maximum")
      left       <- toDoubleArray(a)
      right      <- toDoubleArray(b)
      out        <- createFromFlat(left.zip(right).map((l, r) => math.max(l, r)), leftShape, "maximum")
    yield out

  def diagonal(a: INDArray): Task[INDArray] =
    for
      targetShape <- shape(a)
      out <-
        if targetShape.length != 2 then fail(s"diagonal requires a 2D tensor, got rank ${targetShape.length}")
        else
          for
            values <- toDoubleArray(a)
            rows = targetShape(0).toInt
            cols = targetShape(1).toInt
            diagSize = math.min(rows, cols)
            diagonalValues = Array.tabulate(diagSize)(i => values(i * cols + i))
            out <- createDouble1D(diagonalValues)
          yield out
    yield out

  def solve(a: INDArray, b: INDArray): Task[INDArray] =
    for
      coeffShape <- shape(a)
      rhsShape   <- shape(b)
      out <-
        if coeffShape.length != 2 || coeffShape(0) != coeffShape(1) then fail(s"solve requires a square 2D coefficient matrix, got ${coeffShape.mkString("[", ", ", "]")}")
        else if rhsShape.length == 1 && rhsShape(0) == coeffShape(0) then
          for
            coeffValues <- toDoubleArray(a)
            rhsValues   <- toDoubleArray(b)
            n = coeffShape(0).toInt
            coeffMatrix = Array.tabulate(n, n)((r, c) => coeffValues(r * n + c))
            rhsMatrix = Array.tabulate(n, 1)((r, _) => rhsValues(r))
            solved = solveLinearSystem(coeffMatrix, rhsMatrix).map(_.head)
            out <- createDouble1D(solved)
          yield out
        else if rhsShape.length == 2 && rhsShape(0) == coeffShape(0) then
          for
            coeffValues <- toDoubleArray(a)
            rhsValues   <- toDoubleArray(b)
            n = coeffShape(0).toInt
            rhsCols = rhsShape(1).toInt
            coeffMatrix = Array.tabulate(n, n)((r, c) => coeffValues(r * n + c))
            rhsMatrix = Array.tabulate(n, rhsCols)((r, c) => rhsValues(r * rhsCols + c))
            solved = solveLinearSystem(coeffMatrix, rhsMatrix)
            out <- createDouble(solved)
          yield out
        else fail(s"solve requires RHS rows to match coefficient size ${coeffShape(0)}, got ${rhsShape.mkString("[", ", ", "]")}")
    yield out

  def get(a: INDArray, indices: Array[Long]): Task[Double] =
    for
      targetShape <- shape(a)
      values      <- toDoubleArray(a)
      out <- (targetShape.length, indices.length) match
        case (1, 1) =>
          val idx = indices(0).toInt
          if idx < 0 || idx >= values.length then fail(s"get index $idx out of bounds for length ${values.length}")
          else ZIO.succeed(values(idx))
        case (2, 2) =>
          val row = indices(0).toInt
          val col = indices(1).toInt
          val rows = targetShape(0).toInt
          val cols = targetShape(1).toInt
          if row < 0 || row >= rows || col < 0 || col >= cols then fail(s"get indices (${row}, ${col}) out of bounds for shape ${targetShape.mkString("[", ", ", "]")}")
          else ZIO.succeed(values(row * cols + col))
        case _ => fail(s"get requires indices matching tensor rank ${targetShape.length}, received ${indices.length}")
    yield out

  def slice(a: INDArray, start: Array[Long], end: Array[Long]): Task[INDArray] =
    for
      targetShape <- shape(a)
      values      <- toDoubleArray(a)
      out <- targetShape.length match
        case 1 =>
          if start.length != 1 || end.length != 1 then fail("slice on a 1D tensor requires one start and one end bound")
          else
            for
              bounds <- normalizedBound(start(0), end(0), targetShape(0), 0, "slice")
              (from, until) = bounds
              out <- createDouble1D(values.slice(from, until))
            yield out
        case 2 =>
          if start.length != 2 || end.length != 2 then fail("slice on a 2D tensor requires two start and two end bounds")
          else
            for
              rowBounds <- normalizedBound(start(0), end(0), targetShape(0), 0, "slice")
              colBounds <- normalizedBound(start(1), end(1), targetShape(1), 1, "slice")
              (rowFrom, rowUntil) = rowBounds
              (colFrom, colUntil) = colBounds
              cols = targetShape(1).toInt
              matrix = Array.tabulate(targetShape(0).toInt, cols)((r, c) => values(r * cols + c))
              sliced = Array.tabulate(rowUntil - rowFrom, colUntil - colFrom)((r, c) => matrix(rowFrom + r)(colFrom + c))
              out <- createDouble(sliced)
            yield out
        case _ => fail(s"slice currently supports only 1D or 2D tensors, got rank ${targetShape.length}")
    yield out

  def concatenate(a: INDArray, b: INDArray, axis: Int): Task[INDArray] =
    for
      leftShape  <- shape(a)
      rightShape <- shape(b)
      left       <- toDoubleArray(a)
      right      <- toDoubleArray(b)
      out <- leftShape.length match
        case 1 =>
          if rightShape.length != 1 then fail("concatenate requires tensors of the same rank")
          else if axis != 0 then fail(s"concatenate axis $axis is invalid for 1D tensors")
          else createDouble1D(left ++ right)
        case 2 =>
          if rightShape.length != 2 then fail("concatenate requires tensors of the same rank")
          else if axis == 0 then
            if leftShape(1) != rightShape(1) then fail(s"concatenate axis 0 requires matching column counts: ${leftShape(1)} vs ${rightShape(1)}")
            else createFromFlat(left ++ right, Array(leftShape(0) + rightShape(0), leftShape(1)), "concatenate")
          else if axis == 1 then
            if leftShape(0) != rightShape(0) then fail(s"concatenate axis 1 requires matching row counts: ${leftShape(0)} vs ${rightShape(0)}")
            else
              val rows = leftShape(0).toInt
              val leftCols = leftShape(1).toInt
              val rightCols = rightShape(1).toInt
              val merged = Array.tabulate(rows, leftCols + rightCols) { (r, c) =>
                if c < leftCols then left(r * leftCols + c) else right(r * rightCols + (c - leftCols))
              }
              createDouble(merged)
          else fail(s"concatenate axis $axis is invalid for 2D tensors")
        case _ => fail(s"concatenate currently supports only 1D or 2D tensors, got rank ${leftShape.length}")
    yield out

  def gather(a: INDArray, indices: Array[Int]): Task[INDArray] =
    for
      targetShape <- shape(a)
      _ <- if targetShape.length == 1 then ZIO.unit else fail(s"gather currently supports only 1D tensors, got rank ${targetShape.length}")
      values <- toDoubleArray(a)
      _ <-
        if indices.forall(i => i >= 0 && i < values.length) then ZIO.unit
        else fail(s"gather indices out of bounds for length ${values.length}: ${indices.mkString("[", ", ", "]")}")
      out <- createDouble1D(indices.map(values(_).toDouble))
    yield out

  def lessThanOrEqual(a: INDArray, b: INDArray): Task[INDArray] =
    for
      leftShape  <- shape(a)
      rightShape <- shape(b)
      _          <- requireSameShape(leftShape, rightShape, "lessThanOrEqual")
      left       <- toDoubleArray(a)
      right      <- toDoubleArray(b)
      out        <- createFromFlat(left.zip(right).map((l, r) => if l <= r then 1.0 else 0.0), leftShape, "lessThanOrEqual")
    yield out

  def not(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      targetShape <- shape(a)
      out <- createFromFlat(values.map(v => if math.abs(v) < 1e-12 then 1.0 else 0.0), targetShape, "not")
    yield out

  def where(a: INDArray, mask: INDArray): Task[INDArray] =
    for
      targetShape <- shape(a)
      maskShape   <- shape(mask)
      _ <- requireSameShape(targetShape, maskShape, "where")
      _ <- if targetShape.length == 1 then ZIO.unit else fail(s"where currently supports only 1D tensors, got rank ${targetShape.length}")
      values <- toDoubleArray(a)
      maskValues <- toDoubleArray(mask)
      filtered = values.zip(maskValues).collect { case (value, keep) if math.abs(keep) >= 1e-12 => value }
      out <- createDouble1D(filtered)
    yield out

  def unique(a: INDArray): Task[INDArray] =
    for
      targetShape <- shape(a)
      _ <- if targetShape.length == 1 then ZIO.unit else fail(s"unique currently supports only 1D tensors, got rank ${targetShape.length}")
      values <- toDoubleArray(a)
      out <- createDouble1D(values.distinct)
    yield out

  def countZeros(a: INDArray, threshold: Double = 1e-6): Task[Int] =
    toDoubleArray(a).map(_.count(v => math.abs(v) < threshold))

  def std(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      out <-
        if values.isEmpty then fail("std requires at least one element")
        else
          val meanValue = values.sum / values.length
          val variance = values.map(v => math.pow(v - meanValue, 2)).sum / values.length
          createDouble1D(Array(math.sqrt(variance)))
    yield out

  def norm(a: INDArray): Task[Double] =
    toDoubleArray(a).map(values => math.sqrt(values.map(v => v * v).sum))

  def copy(a: INDArray): Task[INDArray] =
    for
      values <- toDoubleArray(a)
      targetShape <- shape(a)
      out <- createFromFlat(values, targetShape, "copy")
    yield out

  def toFloatArray(a: INDArray): Task[Array[Float]] =
    ZIO.attemptBlocking {
      val arr = new Array[Float](a.length().toInt)
      for i <- arr.indices do arr(i) = a.getFloat(i.toLong)
      arr
    }

  def toDoubleArray(a: INDArray): Task[Array[Double]] =
    ZIO.attemptBlocking {
      val arr = new Array[Double](a.length().toInt)
      for i <- arr.indices do arr(i) = a.getDouble(i.toLong)
      arr
    }

  def shape(a: INDArray): Task[Array[Long]] =
    ZIO.attemptBlocking(a.shape().map(_.toLong))
