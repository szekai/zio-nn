package zio.nn

/** Framework-agnostic preprocessing types for tokenization and image transforms.
  *
  * These are pure data — no framework dependencies, no runtime.
  * Backends implement the actual processing logic.
  */

// ═══════════════════════════════════════════════
//  Tokenizer Configuration
// ═══════════════════════════════════════════════

/** Shared configuration for text tokenization across all backends. */
case class TokenizerConfig(
  maxLength: Int = 512,
  padding: Boolean = true,
  truncation: Boolean = true,
  addSpecialTokens: Boolean = true
)

/** Result of encoding a single text into token IDs.
  *
  * @param tokenIds      token index sequence (fed to predictInt / fitInt)
  * @param attentionMask optional mask indicating real tokens (1) vs padding (0)
  * @param tokenTypeIds  optional segment IDs for NSP-style tasks
  */
case class EncodingResult(
  tokenIds: Array[Int],
  attentionMask: Option[Array[Int]] = None,
  tokenTypeIds: Option[Array[Int]] = None
)

/** Tokenizer backend definitions — user code picks one, backends resolve.
  *
  * == DJL Path ==
  * {{{
  *   val tok = ZTokenizer.huggingFace("bert-base-uncased", config)
  *   val ids = tok.encode("hello world").get.tokenIds  // Array[Int]
  *   model.fitInt(batchOfTokenIds, labels, epochs=5)
  * }}}
  *
  * == DL4J Path ==
  * {{{
  *   val tok = ZTokenizer.regex("\\\\W+", config)
  *   val ids = tok.encode("hello world").get.tokenIds
  * }}}
  */
enum TokenizerDef:
  case HuggingFaceModel(modelName: String, config: TokenizerConfig = TokenizerConfig())
  case Simple(config: TokenizerConfig = TokenizerConfig())

// ═══════════════════════════════════════════════
//  Image Transform Definitions
// ═══════════════════════════════════════════════

/** Image transform specification — framework-agnostic.
  * Compose multiple transforms via `ImagePipeline`.
  *
  * {{{
  *   val pipeline = ImagePipeline(
  *     Resize(224, 224),
  *     Normalize(Array(0.485f, 0.456f, 0.406f), Array(0.229f, 0.224f, 0.225f)),
  *     CenterCrop(200, 200)
  *   )
  * }}}
  */
enum ImageTransformDef:
  /** Resize to target (height, width) pixels. */
  case Resize(height: Int, width: Int)
  /** Normalize with per-channel (mean, std). Both arrays must have same length. */
  case Normalize(mean: Array[Float], std: Array[Float])
  /** Center-crop to target (height, width) pixels. */
  case CenterCrop(height: Int, width: Int)

/** Composable image transform pipeline — pure data, no runtime. */
case class ImagePipeline(transforms: List[ImageTransformDef])

object ImagePipeline:
  def apply(transforms: ImageTransformDef*): ImagePipeline = ImagePipeline(transforms.toList)
