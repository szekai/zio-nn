package zio.nn.rag

import zio.*
import zio.json.*

/** Interface for converting text to embedding vectors.
  *
  * Embeddings are dense vector representations of text that capture semantic meaning.
  * They enable similarity search by allowing documents with similar content to have
  * similar vector representations, which can be compared using cosine distance or
  * inner product.
  */
trait EmbeddingService:
  /** Embed a single text string into a float vector. */
  def embed(text: String): Task[Array[Float]]

  /** Embed multiple texts in a single batch call (more efficient than separate calls). */
  def embedBatch(texts: Chunk[String]): Task[Chunk[Array[Float]]]
end EmbeddingService

object EmbeddingService:
  /** Layer that wires an [[OpenAIEmbeddingService]] as the [[EmbeddingService]]. */
  val layer: ZLayer[OpenAIEmbeddingService, Nothing, EmbeddingService] =
    ZLayer.fromFunction(identity[OpenAIEmbeddingService])
end EmbeddingService

/** Minimal `/v1/embeddings` request body for the OpenAI API. */
@jsonMemberNames(SnakeCase)
case class OpenAIEmbeddingRequest(model: String, input: List[String])

object OpenAIEmbeddingRequest:
  given JsonCodec[OpenAIEmbeddingRequest] = DeriveJsonCodec.gen

/** A single embedding result returned by the OpenAI API. */
@jsonMemberNames(SnakeCase)
case class OpenAIEmbeddingData(embedding: List[Float], index: Int)

object OpenAIEmbeddingData:
  given JsonCodec[OpenAIEmbeddingData] = DeriveJsonCodec.gen

/** Top-level `/v1/embeddings` response from the OpenAI API. */
@jsonMemberNames(SnakeCase)
case class OpenAIEmbeddingResponse(data: List[OpenAIEmbeddingData], model: String)

object OpenAIEmbeddingResponse:
  given JsonCodec[OpenAIEmbeddingResponse] = DeriveJsonCodec.gen

/** OpenAI-compatible embedding service using the `/v1/embeddings` endpoint.
  *
  * Defaults to `text-embedding-ada-002` with a configurable base URL for
  * self-hosted or alternative OpenAI-compatible providers.
  *
  * @param apiKey  OpenAI API key (or compatible provider key)
  * @param baseUrl API base URL, defaults to `https://api.openai.com/v1`
  */
class OpenAIEmbeddingService(apiKey: String, baseUrl: String = "https://api.openai.com/v1")
  extends EmbeddingService:

  private val model = "text-embedding-ada-002"

  override def embed(text: String): Task[Array[Float]] =
    embedBatch(Chunk(text)).map(_.head)

  override def embedBatch(texts: Chunk[String]): Task[Chunk[Array[Float]]] =
    val requestBody = OpenAIEmbeddingRequest(model, texts.toList)
    for
      jsonStr  <- ZIO.succeed(requestBody.toJson)
      response <- httpPost(jsonStr)
      parsed   <- ZIO.fromEither(response.fromJson[OpenAIEmbeddingResponse])
        .mapError(e => new RuntimeException(s"JSON parse error: $e"))
      result   =  parsed.data.sortBy(_.index).map(_.embedding.toArray)
    yield Chunk.fromIterable(result)

  private def httpPost(json: String): Task[String] =
    import java.net.http.{HttpClient, HttpRequest, HttpResponse}
    import java.net.URI
    import java.time.Duration

    for
      client <- ZIO.attempt(HttpClient.newBuilder.connectTimeout(Duration.ofSeconds(30)).build())
      req    <- ZIO.attempt(
        HttpRequest.newBuilder
          .uri(URI.create(s"$baseUrl/embeddings"))
          .header("Content-Type", "application/json")
          .header("Authorization", s"Bearer $apiKey")
          .POST(HttpRequest.BodyPublishers.ofString(json))
          .timeout(Duration.ofSeconds(60))
          .build()
      )
      resp   <- ZIO.attempt(client.send(req, HttpResponse.BodyHandlers.ofString()))
      body   <- ZIO.attempt {
                  if resp.statusCode() >= 400 then
                    throw new RuntimeException(s"OpenAI API error ${resp.statusCode()}: ${resp.body()}")
                  else resp.body()
                }
    yield body
end OpenAIEmbeddingService
