package zio.nn.dl4j

import zio.nn.{EncodingResult, TokenizerConfig}
import scala.util.Try

/** DL4J-native regex-based tokenizer (no HuggingFace dependency).
  *
  * Falls back to simple whitespace/regex splitting when HuggingFace model
  * tokenizers are unavailable. For production NLP pipelines, use the DJL
  * backend with HuggingFace tokenizers instead.
  *
  * {{{
  *   val tok = ZTokenizer.regex("\\\\W+").get
  *   val ids = tok.encode("hello world").get.tokenIds  // Array[Int]
  *   model.fitInt(batchOfIds, labels, epochs = 5)
  *   tok.close()
  * }}}
  */
class ZTokenizer private (
  val split: String => Array[String],
  val vocabulary: Map[String, Int],
  val config: TokenizerConfig
):

  /** Encode a single text string into token IDs via vocabulary lookup. */
  def encode(text: String): Try[EncodingResult] = Try {
    val tokens = split(text)
    val ids = tokens.flatMap(t => vocabulary.get(t.toLowerCase))
    EncodingResult(
      tokenIds      = ids,
      attentionMask = if config.padding then Some(Array.fill(ids.length)(1)) else None,
      tokenTypeIds  = None
    )
  }

  /** Encode a batch of texts into token IDs. */
  def batchEncode(texts: Array[String]): Try[Array[EncodingResult]] = Try {
    texts.map(encode(_).get)
  }

  /** Decode token IDs back into text via inverse vocabulary lookup. */
  def decode(tokens: Array[Int]): Try[String] = Try {
    val inverse = vocabulary.map(_.swap)
    tokens.flatMap(inverse.get).mkString(" ")
  }

  def close(): Unit = ()

object ZTokenizer:

  /** Create a regex-based tokenizer that splits on the given pattern.
    * A simple character-level vocabulary is built from printable ASCII.
    *
    * @param pattern  regex pattern to split on (default: "\\W+" = non-word chars)
    * @param config   tokenizer configuration
    */
  def regex(pattern: String = "\\W+", config: TokenizerConfig = TokenizerConfig()): Try[ZTokenizer] = Try {
    val vocab = (32 to 126).map(_.toChar.toString).zipWithIndex.toMap
    new ZTokenizer(
      split      = (s: String) => s.split(pattern).filter(_.nonEmpty),
      vocabulary = vocab,
      config     = config
    )
  }

  /** Create a whitespace-splitting tokenizer (splits on spaces/newlines/tabs).
    * A simple character-level vocabulary is built from printable ASCII.
    */
  def whitespace(config: TokenizerConfig = TokenizerConfig()): ZTokenizer =
    ZTokenizer.regex("\\s+", config).get
