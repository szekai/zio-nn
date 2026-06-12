package zio.nn.dl4j

import zio.nn.*
import zio.test.*
import zio.test.Assertion.*

object ZTokenizerSpec extends ZIOSpecDefault:

  def spec = suite("ZTokenizer (DL4J — regex/whitespace)")(
    test("regex tokenizer encodes text into token IDs") {
      val tok = ZTokenizer.regex("\\W+").get
      val result = tok.encode("a b c")
      tok.close()
      assertTrue(result.isSuccess, result.get.tokenIds.length == 3)
    },
    test("batchEncode handles multiple texts") {
      val tok = ZTokenizer.regex("\\W+").get
      val result = tok.batchEncode(Array("a b", "c d e"))
      tok.close()
      assertTrue(result.isSuccess, result.get.length == 2)
    },
    test("decode roundtrip preserves original text") {
      val tok = ZTokenizer.regex("\\W+").get
      val encoded = tok.encode("a b c").get
      val decoded = tok.decode(encoded.tokenIds)
      tok.close()
      assertTrue(decoded.isSuccess, decoded.get == "a b c")
    },
    test("whitespace tokenizer splits on spaces") {
      val tok = ZTokenizer.whitespace()
      val result = tok.encode("a b c")
      tok.close()
      assertTrue(result.isSuccess, result.get.tokenIds.length == 3)
    },
    test("fromVocabulary creates tokenizer with custom vocabulary") {
      val vocab = Map("hello" -> 0, "world" -> 1)
      val tok = ZTokenizer.fromVocabulary(vocab)
      val result = tok.encode("hello world").get
      tok.close()
      assertTrue(result.tokenIds sameElements Array(0, 1))
    },
    test("fromVocabulary maps unknown tokens to unkIndex when set") {
      val vocab = Map("hello" -> 0, "world" -> 1)
      val tok = ZTokenizer.fromVocabulary(vocab, unkIndex = Some(2))
      val result = tok.encode("hello unknown").get
      tok.close()
      assertTrue(result.tokenIds sameElements Array(0, 2))
    },
    test("fromVocabulary drops unknown tokens when unkIndex is None") {
      val vocab = Map("hello" -> 0, "world" -> 1)
      val tok = ZTokenizer.fromVocabulary(vocab, unkIndex = None)
      val result = tok.encode("hello unknown").get
      tok.close()
      assertTrue(result.tokenIds sameElements Array(0))
    }
  )
