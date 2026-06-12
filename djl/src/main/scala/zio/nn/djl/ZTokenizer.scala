package zio.nn.djl

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import zio.nn.{EncodingResult, TokenizerConfig}
import java.nio.file.Path
import scala.util.Try

/** DJL-backed tokenizer wrapping HuggingFaceTokenizer (Rust tokenizers).
  *
  * Primary use case: convert text to token IDs for models with Embedding as first layer.
  * {{{
  *   val tok = ZTokenizer.huggingFace("bert-base-uncased").get
  *   val ids = tok.encode("hello world").get.tokenIds  // Array[Int]
  *   model.fitInt(batchOfIds, labels, epochs = 5)
  *   tok.close()
  * }}}
  */
class ZTokenizer private (underlying: HuggingFaceTokenizer, val config: TokenizerConfig):

  /** Encode a single text string into token IDs. */
  def encode(text: String): Try[EncodingResult] = Try {
    val e = underlying.encode(text)
    EncodingResult(
      tokenIds      = e.getIds.map(_.toInt),
      attentionMask = if config.padding then Some(e.getAttentionMask.map(_.toInt)) else None,
      tokenTypeIds  = if config.addSpecialTokens then Some(e.getTypeIds.map(_.toInt)) else None
    )
  }

  /** Encode a batch of texts into token IDs. */
  def batchEncode(texts: Array[String]): Try[Array[EncodingResult]] = Try {
    val batch = underlying.batchEncode(texts)
    batch.map { e =>
      EncodingResult(
        tokenIds      = e.getIds.map(_.toInt),
        attentionMask = if config.padding then Some(e.getAttentionMask.map(_.toInt)) else None,
        tokenTypeIds  = if config.addSpecialTokens then Some(e.getTypeIds.map(_.toInt)) else None
      )
    }
  }

  /** Decode token IDs back into text. */
  def decode(tokens: Array[Int]): Try[String] = Try {
    underlying.decode(tokens.map(_.toLong))
  }

  def close(): Unit = underlying.close()

object ZTokenizer:

  /** Create from a HuggingFace hub model name (e.g. "bert-base-uncased").
    * Downloads the tokenizer files on first use.
    */
  def huggingFace(modelName: String, config: TokenizerConfig = TokenizerConfig()): Try[ZTokenizer] = Try {
    val ht = HuggingFaceTokenizer.newInstance(modelName)
    new ZTokenizer(ht, config)
  }

  /** Create from a local `tokenizer.json` file path. */
  def fromJson(path: Path, config: TokenizerConfig = TokenizerConfig()): Try[ZTokenizer] = Try {
    val ht = HuggingFaceTokenizer.newInstance(path.toString)
    new ZTokenizer(ht, config)
  }
