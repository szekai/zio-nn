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
    }
  )
