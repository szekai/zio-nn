package zio.nn.storch

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object TensorOpsSpec extends ZIOSpecDefault:

  def spec = suite("TensorOps (storch)")(
    test("zeros creates tensor of correct shape") {
      val t = TensorOps.zeros(Array(3, 4)).get
      val s = t.shape
      assertTrue(s.length == 2, s(0).toInt == 3, s(1).toInt == 4, t.toArray.forall(_ == 0.0f))
    },
    test("ones creates tensor of correct shape") {
      val t = TensorOps.ones(Array(2, 5)).get
      val s = t.shape
      assertTrue(s(0).toInt == 2, s(1).toInt == 5, t.toArray.forall(_ == 1.0f))
    },
    test("add works element-wise") {
      val a = torch.Tensor(Array(1.0f, 2.0f, 3.0f))
      val b = torch.Tensor(Array(10.0f, 20.0f, 30.0f))
      val c = TensorOps.add(a, b).get
      val arr = c.toArray
      assertTrue(arr.toSeq == Seq(11.0f, 22.0f, 33.0f))
    },
    test("sub works element-wise") {
      val a = torch.Tensor(Array(10.0f, 20.0f, 30.0f))
      val b = torch.Tensor(Array(1.0f, 2.0f, 3.0f))
      val c = TensorOps.sub(a, b).get
      val arr = c.toArray
      assertTrue(arr.toSeq == Seq(9.0f, 18.0f, 27.0f))
    },
    test("mul works element-wise") {
      val a = torch.Tensor(Array(2.0f, 3.0f, 4.0f))
      val b = torch.Tensor(Array(5.0f, 6.0f, 7.0f))
      val c = TensorOps.mul(a, b).get
      val arr = c.toArray
      assertTrue(arr.toSeq == Seq(10.0f, 18.0f, 28.0f))
    },
    test("div works element-wise") {
      val a = torch.Tensor(Array(10.0f, 20.0f, 30.0f))
      val b = torch.Tensor(Array(2.0f, 5.0f, 3.0f))
      val c = TensorOps.div(a, b).get
      val arr = c.toArray
      assertTrue(arr.toSeq == Seq(5.0f, 4.0f, 10.0f))
    },
    test("matmul works") {
      val a = torch.Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f)).reshape(2, 2)
      val b = torch.Tensor(Array(5.0f, 6.0f, 7.0f, 8.0f)).reshape(2, 2)
      val c = TensorOps.matmul(a, b).get
      val s = c.shape
      val arr = c.toArray
      assertTrue(s(0).toInt == 2, s(1).toInt == 2,
                 math.abs(arr(0) - 19.0f) < 1e-5,
                 math.abs(arr(1) - 22.0f) < 1e-5,
                 math.abs(arr(2) - 43.0f) < 1e-5,
                 math.abs(arr(3) - 50.0f) < 1e-5)
    },
    test("reshape works") {
      val t = torch.Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
      val r = TensorOps.reshape(t, Array(2, 3)).get
      val s = r.shape
      assertTrue(s(0).toInt == 2, s(1).toInt == 3)
    },
    test("transpose works") {
      val t = torch.Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f)).reshape(2, 2)
      val tr = TensorOps.transpose(t, Array(0, 1)).get
      val s = tr.shape
      assertTrue(s(0).toInt == 2, s(1).toInt == 2)
    },
    test("squeeze and unsqueeze work") {
      val t = torch.Tensor(Array(1.0f, 2.0f, 3.0f)).reshape(3, 1)
      val sq = TensorOps.squeeze(t, dim = 1).get
      assertTrue(sq.shape.length == 1, sq.shape(0).toInt == 3)
      val usq = TensorOps.unsqueeze(sq, dim = 0).get
      assertTrue(usq.shape.length == 2, usq.shape(0).toInt == 1)
    },
    test("stack works") {
      val a = torch.Tensor(Array(1.0f, 2.0f))
      val b = torch.Tensor(Array(3.0f, 4.0f))
      val s = TensorOps.stack(Seq(a, b), dim = 0).get
      assertTrue(s.shape(0).toInt == 2, s.shape(1).toInt == 2)
    },
    test("concat works") {
      val a = torch.Tensor(Array(1.0f, 2.0f)).reshape(1, 2)
      val b = torch.Tensor(Array(3.0f, 4.0f)).reshape(1, 2)
      val c = TensorOps.concat(Seq(a, b), dim = 0).get
      assertTrue(c.shape(0).toInt == 2, c.shape(1).toInt == 2)
    },
    test("sum reduces correctly") {
      val t = torch.Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f)).reshape(2, 2)
      val s = TensorOps.sum(t).get
      val arr = s.toArray
      assertTrue(s.shape.length == 0 || s.shape.length == 1,
                 math.abs(arr(0) - 10.0f) < 1e-5)
    },
    test("mean reduces correctly") {
      val t = torch.Tensor(Array(2.0f, 4.0f, 6.0f, 8.0f)).reshape(2, 2)
      val m = TensorOps.mean(t).get
      val arr = m.toArray
      assertTrue(math.abs(arr(0) - 5.0f) < 1e-5)
    },
    test("gather works on classification indices") {
      val t = torch.Tensor(Array(0.1f, 0.2f, 0.7f, 0.3f, 0.4f, 0.3f)).reshape(2, 3)
      val idx = torch.Tensor(Array(2L, 1L)).reshape(2, 1)
      val g = TensorOps.gather(t, dim = 1, index = idx).get
      val arr = g.toArray
      assertTrue(math.abs(arr(0) - 0.7f) < 1e-5,
                 math.abs(arr(1) - 0.4f) < 1e-5)
    },
    test("toArray round-trips correctly") {
      val original = Array(1.5f, 2.5f, 3.5f)
      val t = torch.Tensor(original)
      val result = t.toArray
      assertTrue(result.toSeq == original.toSeq)
    }
  )
