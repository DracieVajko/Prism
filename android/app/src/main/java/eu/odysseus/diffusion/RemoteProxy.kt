package eu.odysseus.diffusion

import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.request.headers
import io.ktor.client.request.request
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.readBytes
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.httpMethod
import io.ktor.server.request.receiveChannel
import io.ktor.server.request.uri
import io.ktor.server.response.respondBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import android.util.Log
import java.io.ByteArrayOutputStream

/**
 * Forwards requests to a remote PC server when the app is in remote mode.
 * Used when AppConfig.connectionMode is "remote" or "auto" (fallback).
 */
object RemoteProxy {
    private const val TAG = "RemoteProxy"
    private val client = HttpClient(OkHttp) {
        engine {
            config {
                followRedirects(false)
            }
        }
    }

    /** Check if remote mode is active and configured. */
    fun isRemoteMode(): Boolean =
        AppConfig.connectionMode == "remote" && AppConfig.remoteHost.isNotBlank()

    /** Check if auto mode is active and configured. */
    fun isAutoMode(): Boolean =
        AppConfig.connectionMode == "auto" && AppConfig.remoteHost.isNotBlank()

    /** Read all bytes from a ByteReadChannel. */
    private suspend fun readAllBytes(channel: io.ktor.utils.io.ByteReadChannel): ByteArray {
        val buffer = ByteArrayOutputStream()
        val buf = java.nio.ByteBuffer.allocate(8192)
        while (true) {
            val read = channel.readAvailable(buf)
            if (read == -1) break
            if (read > 0) {
                buf.flip()
                val bytes = ByteArray(read)
                buf.get(bytes)
                buffer.write(bytes)
                buf.clear()
            }
        }
        return buffer.toByteArray()
    }

    /** Forward a request to the remote PC server. Returns true if forwarded. */
    suspend fun forward(call: ApplicationCall): Boolean {
        val host = AppConfig.remoteHost.trim().trimEnd('/')
        if (host.isBlank()) return false
        if (!host.startsWith("http://") && !host.startsWith("https://")) {
            return false
        }
        val url = host + call.request.uri
        return try {
            val method = call.request.httpMethod
            val reqHeaders = call.request.headers
            val body = if (method == HttpMethod.Get || method == HttpMethod.Delete) {
                null
            } else {
                readAllBytes(call.request.receiveChannel())
            }

            val resp: HttpResponse = withContext(Dispatchers.IO) {
                client.request(url) {
                    this.method = method
                    headers {
                        reqHeaders.forEach { key, values ->
                            if (key.equals(HttpHeaders.Host, ignoreCase = true)) return@forEach
                            if (key.equals(HttpHeaders.ContentLength, ignoreCase = true)) return@forEach
                            if (key.equals(HttpHeaders.Connection, ignoreCase = true)) return@forEach
                            values.forEach { v -> append(key, v) }
                        }
                    }
                    if (body != null && body.isNotEmpty()) {
                        setBody(body)
                    }
                }
            }

            val respBytes = resp.readBytes()
            val contentType = resp.headers[HttpHeaders.ContentType]
            call.respondBytes(respBytes, contentType = contentType?.let {
                ContentType.parse(it)
            }, status = resp.status)
            true
        } catch (e: Exception) {
            Log.w(TAG, "forward failed: ${e.message}", e)
            false
        }
    }
}
