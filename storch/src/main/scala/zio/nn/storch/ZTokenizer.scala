package zio.nn.storch

import zio.nn.{EncodingResult, TokenizerConfig}
import zio.{Task, ZIO}
import scala.util.Try

/** Simple word-level tokenizer for the storch backend.
  *
  * Splits text on whitespace and punctuation using regex,
  * maps tokens to integer indices via a vocabulary,
  * and produces [[EncodingResult]] with attention masks.
  *
  * Since storch (bytedeco/storch) has no built-in HuggingFace tokenizer support,
  * this mirrors the DL4J approach of local regex-based tokenization.
  *
  * {{{
  *   val vocab = Map("hello" -> 0, "world" -> 1)
  *   val config = TokenizerConfig(maxLength = 16)
  *   val result = ZTokenizer.encode("hello world", vocab, config)
  *   // result.tokenIds == Array(0, 1, 0, 0, ...)  (padded)
  *   // result.attentionMask == Some(Array(1, 1, 0, 0, ...))
  * }}}
  */
object ZTokenizer:

  @volatile
  private var reverseVocab: Option[Map[Int, String]] = None

  private def getOrBuildReverseVocab(vocab: Map[String, Int]): Map[Int, String] =
    reverseVocab match
      case Some(cached) => cached
      case None =>
        val rev = vocab.map(_.swap)
        reverseVocab = Some(rev)
        rev

  /** Encode a single text string into token IDs + attention mask.
    *
    * Splits on non-word characters (whitespace, punctuation), lowercases each token,
    * looks up in `vocab` (falling back to `unkIndex`), truncates to
    * `config.maxLength` when `config.truncation` is true, and pads to
    * `config.maxLength` with `padIndex` when `config.padding` is true.
    *
    * @param text      input text to tokenize
    * @param vocab     token-to-index vocabulary
    * @param config    tokenizer configuration (maxLength, padding, truncation)
    * @param unkIndex  index to use for out-of-vocabulary tokens (default: 1)
    * @param padIndex  index to use for padding tokens (default: 0)
    * @return [[EncodingResult]] with token IDs and optional attention mask
    */
  def encode(
    text: String,
    vocab: Map[String, Int],
    config: TokenizerConfig,
    unkIndex: Int = 1,
    padIndex: Int = 0
  ): EncodingResult =
    val tokens = text.split("\\W+").filter(_.nonEmpty)
    val truncated = if config.truncation then tokens.take(config.maxLength) else tokens
    val ids = truncated.map(t => vocab.getOrElse(t.toLowerCase, unkIndex))
    val padded =
      if config.padding then ids.padTo(config.maxLength, padIndex) else ids
    val attentionMask =
      if config.padding then
        Some(Array.tabulate(config.maxLength)(i => if i < ids.length then 1 else 0))
      else None
    EncodingResult(tokenIds = padded, attentionMask = attentionMask)

  /** Encode a text string, returning only token IDs wrapped in [[Try]]. */
  def encodeTry(
    text: String,
    vocab: Map[String, Int],
    config: TokenizerConfig,
    unkIndex: Int = 1,
    padIndex: Int = 0
  ): Try[Array[Int]] = Try {
    encode(text, vocab, config, unkIndex, padIndex).tokenIds
  }

  /** Encode a text string, returning only token IDs wrapped in [[Task]]. */
  def encodeZ(
    text: String,
    vocab: Map[String, Int],
    config: TokenizerConfig,
    unkIndex: Int = 1,
    padIndex: Int = 0
  ): Task[Array[Int]] = ZIO.attempt {
    encode(text, vocab, config, unkIndex, padIndex).tokenIds
  }

  /** Decode token IDs back into text via inverse vocabulary lookup.
    *
    * Strips padding tokens (`padIndex`) before joining the remaining tokens
    * with a single space.
    *
    * @param tokenIds  token index sequence to decode
    * @param vocab     token-to-index vocabulary (used to build reverse map)
    * @param padIndex  padding token index to strip from output (default: 0)
    * @return decoded text string
    */
  def decode(
    tokenIds: Array[Int],
    vocab: Map[String, Int],
    padIndex: Int = 0
  ): String =
    val rev = getOrBuildReverseVocab(vocab)
    tokenIds.filter(_ != padIndex).flatMap(rev.get).mkString(" ")

  /** Encode a batch of texts into token ID arrays.
    *
    * @param texts     sequence of input texts
    * @param vocab     token-to-index vocabulary
    * @param config    tokenizer configuration
    * @param unkIndex  index for unknown tokens (default: 1)
    * @param padIndex  index for padding tokens (default: 0)
    * @return array of token ID arrays, one per input text
    */
  def encodeBatch(
    texts: Seq[String],
    vocab: Map[String, Int],
    config: TokenizerConfig,
    unkIndex: Int = 1,
    padIndex: Int = 0
  ): Array[Array[Int]] =
    texts.map(t => encode(t, vocab, config, unkIndex, padIndex).tokenIds).toArray
