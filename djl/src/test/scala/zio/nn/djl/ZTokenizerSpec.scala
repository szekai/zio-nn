package zio.nn.djl

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object ZTokenizerSpec extends ZIOSpecDefault:

  def spec = suite("ZTokenizer (DJL)")(
    test("huggingFace returns Failure for non-existent model") {
      val result = ZTokenizer.huggingFace("this-model-does-not-exist-12345")
      assertTrue(result.isFailure)
    },
    test("fromJson returns Failure for non-existent path") {
      val result = ZTokenizer.fromJson(java.nio.file.Paths.get("/nonexistent/tokenizer.json"))
      assertTrue(result.isFailure)
    },
    test("encode returns non-empty token IDs") {
      val tok = ZTokenizer.huggingFace("bert-base-uncased").get
      try
        val result = tok.encode("hello world").get
        assertTrue(result.tokenIds.nonEmpty)
      finally tok.close()
    } @@ TestAspect.ignore,
    test("encode includes attention mask when padding is true") {
      val tok = ZTokenizer.huggingFace("bert-base-uncased").get
      try
        val result = tok.encode("hello world").get
        assertTrue(result.attentionMask.isDefined, result.attentionMask.get.nonEmpty)
      finally tok.close()
    } @@ TestAspect.ignore,
    test("encode includes token type IDs when addSpecialTokens is true") {
      val tok = ZTokenizer.huggingFace("bert-base-uncased").get
      try
        val result = tok.encode("hello world").get
        assertTrue(result.tokenTypeIds.isDefined, result.tokenTypeIds.get.nonEmpty)
      finally tok.close()
    } @@ TestAspect.ignore,
    test("batchEncode handles multiple texts") {
      val tok = ZTokenizer.huggingFace("bert-base-uncased").get
      try
        val results = tok.batchEncode(Array("hello", "world")).get
        assertTrue(results.length == 2, results.forall(_.tokenIds.nonEmpty))
      finally tok.close()
    } @@ TestAspect.ignore,
    test("decode roundtrip preserves meaning") {
      val tok = ZTokenizer.huggingFace("bert-base-uncased").get
      try
        val encoded = tok.encode("hello world").get
        val decoded = tok.decode(encoded.tokenIds).get
        assertTrue(decoded.toLowerCase.contains("hello"), decoded.toLowerCase.contains("world"))
      finally tok.close()
    } @@ TestAspect.ignore
  )
