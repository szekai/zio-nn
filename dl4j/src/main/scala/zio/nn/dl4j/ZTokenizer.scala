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
  *
  * @param unkIndex  when set, unknown tokens are mapped to this index instead of dropped
  */
class ZTokenizer private (
  val split: String => Array[String],
  val vocabulary: Map[String, Int],
  val config: TokenizerConfig,
  val unkIndex: Option[Int] = None
):

  /** Encode a single text string into token IDs via vocabulary lookup.
    * When `unkIndex` is set, tokens not in the vocabulary get that index.
    * Otherwise, unknown tokens are silently dropped.
    */
  def encode(text: String): Try[EncodingResult] = Try {
    val tokens = split(text)
    val ids = unkIndex match
      case Some(unk) => tokens.map(t => vocabulary.getOrElse(t.toLowerCase, unk))
      case None      => tokens.flatMap(t => vocabulary.get(t.toLowerCase))
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
    * @param pattern    regex pattern to split on (default: "\\W+" = non-word chars)
    * @param config     tokenizer configuration
    * @param vocabRange range of character codes to build vocabulary from (default: 32 to 126)
    */
  def regex(pattern: String = "\\W+", config: TokenizerConfig = TokenizerConfig(), vocabRange: Range = 32 to 126): Try[ZTokenizer] = Try {
    val vocab = vocabRange.map(_.toChar.toString).zipWithIndex.toMap
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

  /** Create a tokenizer from an explicit vocabulary map.
    *
    * Useful for bootstrapping a tokenizer from pre-trained model vocabularies
    * (e.g. Word2Vec, GloVe). Each token in the input text is split by the
    * `split` function, then looked up in `vocabulary`. When `unkIndex` is set,
    * out-of-vocabulary tokens get that index instead of being dropped.
    *
    * @param vocabulary  word → token ID mapping
    * @param split       function to split text into tokens (default: whitespace)
    * @param config      tokenizer configuration
    * @param unkIndex    index for unknown tokens (None = drop unknowns)
    */
  def fromVocabulary(
    vocabulary: Map[String, Int],
    split: String => Array[String] = _.split("\\s+").filter(_.nonEmpty),
    config: TokenizerConfig = TokenizerConfig(),
    unkIndex: Option[Int] = None
  ): ZTokenizer =
    new ZTokenizer(split, vocabulary, config, unkIndex)
