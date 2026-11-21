package eu.odysseus.diffusion

import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Minimal OpenAI-compatible LLM client for Android post-processing.
 *
 * Supports: OpenAI, Gemini (via OpenAI-compat), Mistral, Ollama, LM Studio,
 * and any custom endpoint that speaks /v1/chat/completions.
 */
object LlmClient {
    private const val TAG = "LlmClient"
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    /**
     * Generate a completion. Returns null on failure.
     */
    fun complete(prompt: String, systemPrompt: String = ""): String? {
        val baseUrl = AppConfig.llmBaseUrl.trim().trimEnd('/')
        val model = AppConfig.llmModel.trim()
        val apiKey = AppConfig.llmApiKey.trim()

        if (baseUrl.isBlank() || model.isBlank()) return null

        val url = if (baseUrl.endsWith("/v1")) "$baseUrl/chat/completions"
                  else "$baseUrl/v1/chat/completions"

        val messages = JSONArray()
        if (systemPrompt.isNotBlank()) {
            messages.put(JSONObject().put("role", "system").put("content", systemPrompt))
        }
        messages.put(JSONObject().put("role", "user").put("content", prompt))

        val body = JSONObject()
            .put("model", model)
            .put("messages", messages)
            .put("max_tokens", 1024)
            .put("temperature", 0.3)
            .toString()

        val reqBuilder = Request.Builder().url(url)
            .post(body.toRequestBody("application/json".toMediaType()))
        if (apiKey.isNotBlank()) {
            reqBuilder.header("Authorization", "Bearer $apiKey")
        }

        return try {
            val resp = client.newCall(reqBuilder.build()).execute()
            if (!resp.isSuccessful) {
                Log.w(TAG, "HTTP ${resp.code}: ${resp.body?.string()?.take(200)}")
                return null
            }
            val json = JSONObject(resp.body!!.string())
            json.getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
                .trim()
        } catch (e: Exception) {
            Log.w(TAG, "complete failed: ${e.message}", e)
            null
        }
    }

    /**
     * Chunk text and process each chunk through the LLM.
     */
    fun processChunks(text: String, transform: (String) -> String): String {
        val maxInputChars = (AppConfig.llmMaxContextTokens - AppConfig.llmOutputReserve) * 4 / 5
        val chunks = mutableListOf<String>()
        var remaining = text
        while (remaining.length > maxInputChars) {
            var cut = remaining.lastIndexOf(". ", maxInputChars.toInt())
            if (cut < maxInputChars.toInt() / 2) cut = remaining.lastIndexOf(" ", maxInputChars.toInt())
            if (cut < maxInputChars.toInt() / 2) cut = maxInputChars.toInt()
            chunks.add(remaining.substring(0, cut + 1).trim())
            remaining = remaining.substring(cut + 1).trim()
        }
        if (remaining.isNotEmpty()) chunks.add(remaining)
        if (chunks.isEmpty()) chunks.add(text)

        val results = mutableListOf<String>()
        for (chunk in chunks) {
            val result = transform(chunk)
            results.add(result ?: chunk)
        }
        return results.joinToString("\n\n")
    }
}
