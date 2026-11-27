package eu.dracievajko.lenswright

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.util.Base64
import android.util.Log
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.client.plugins.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.ArrayDeque
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * Owns a live transcription session on the phone: microphone and (optional)
 * device-internal audio capture, two realtime WebSockets, and the HTTP appends
 * that feed [LiveSessionStore].
 *
 * Everything talks to the local Ktor server only (`127.0.0.1:8101`); the
 * live session routes and `/v1/realtime` are never forwarded to a PC. Only
 * transcript text ever reaches the server/disk - no PCM bytes are written to
 * any file. Capture runs continuously on [Dispatchers.IO] and never blocks on
 * the network: a bounded queue drops the oldest unprocessed chunk on overflow.
 *
 * The drain contract is exercised by a plain JVM unit test, so the object has
 * no Android-framework work at class-initialisation time and
 * [awaitPending]/[setPendingForTest]/[completePendingForTest]/[status] are
 * pure-JVM code paths built only on a lock/condition and an integer counter.
 */
object LiveCapture {
    private const val TAG = "LiveCapture"

    // Local-only server. Live endpoints are never forwarded (see ApiServer).
    private const val HOST = "127.0.0.1"
    private const val PORT = 8101
    private const val WS_URL = "ws://$HOST:$PORT/v1/realtime"
    private const val HTTP_BASE = "http://$HOST:$PORT"

    // The only two transcript sources.
    private const val SOURCE_MIC = "mic"
    private const val SOURCE_DEVICE = "device"

    // 16 kHz mono PCM16, matching the WebUI realtime client.
    private const val SAMPLE_RATE_HZ = 16000
    private const val BYTES_PER_SAMPLE = 2
    private const val COMMIT_MS = 5000L
    private const val READ_SIZE = SAMPLE_RATE_HZ * BYTES_PER_SAMPLE / 10 // 100 ms

    // At most two commits in flight per source; at most two waiting chunks.
    private const val MAX_INFLIGHT = 2
    private const val MAX_QUEUE = 2

    // Bounded wait for in-flight commits on Stop, before the session is closed.
    private const val DRAIN_TIMEOUT_MS = 5000L

    // Outstanding chunks = queued + in-flight, across all sources.
    // awaitPending() waits for this to reach zero. It is guarded by
    // [pendingLock] and is the single, JVM-safe piece of shared state.
    // ReentrantLock/Condition are used instead of Any.wait/notify because
    // Kotlin's Any does not expose the java.lang.Object monitor methods.
    private val pendingLock = java.util.concurrent.locks.ReentrantLock()
    private val pendingChanged = pendingLock.newCondition()
    private var outstanding = 0

    /** Run [block] while holding [pendingLock]. */
    private inline fun <T> withPending(block: () -> T): T {
        pendingLock.lock()
        try {
            return block()
        } finally {
            pendingLock.unlock()
        }
    }

    /** Wake every thread waiting in [awaitPending]. Call with [pendingLock] held. */
    private fun signalPending() = pendingChanged.signalAll()

    @Volatile private var running = false
    @Volatile private var loss = false
    @Volatile private var sessionId: String = ""
    private var startMs = 0L
    private var appContext: Context? = null
    @Volatile private var projection: MediaProjection? = null
    @Volatile private var sourceSkipReasons: Map<String, String> = emptyMap()

    // Active sources, keyed by source name. Concurrent because Stop/Start and
    // the IO coroutines may touch it from different threads.
    private val sources = ConcurrentHashMap<String, SourceState>()

    // Lazily built so that loading this object in a local JVM unit test never
    // touches the Android framework or opens a socket/client.
    private val wsClient: HttpClient by lazy {
        HttpClient(OkHttp) { install(WebSockets) }
    }
    private val postClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            // Total-call timeout bounds the entire append/close POST (connect,
            // write, read) so a stalled localhost call cannot extend teardown.
            // append POSTs run on IO coroutines; the close POST runs during the
            // off-main stop() teardown.
            .callTimeout(5, TimeUnit.SECONDS)
            .build()
    }
    private val ioScope: CoroutineScope by lazy {
        CoroutineScope(SupervisorJob() + Dispatchers.IO)
    }

    /** One 5-second (or final partial) chunk of raw PCM16 bytes. */
    private data class Chunk(val pcm: ByteArray, val elapsedMs: Long)

    /** Per-source capture, queue and socket state. */
    private class SourceState(val source: String, val sid: String) {
        val queue = ArrayDeque<Chunk>()
        val awaitingElapsed = ArrayDeque<Long>()
        val itemElapsed = HashMap<String, Long>()
        // A single consumer serialises append+commit pairs onto the socket.
        val outbox = Channel<Chunk>(Channel.UNLIMITED)
        val buffer = ByteArrayOutputStream()
        @Volatile var inFlight = 0
        @Volatile var socketOpen = false
        @Volatile var lastCutMs = 0L
        var record: AudioRecord? = null
        var captureJob: kotlinx.coroutines.Job? = null
        var socketJob: kotlinx.coroutines.Job? = null
    }

    // --------------------------------------------------------------- public API

    /** Store the application context for the service lifetime. */
    fun attach(ctx: Context) {
        appContext = ctx.applicationContext
    }

    /**
     * Start capturing [sources] ("mic" and/or "device") into session
     * [sessionId].
     *
     * The session is created (by the UI through `POST /v1/live/sessions`) and
     * passed in here. Because [LiveSessionStore.markSocket] throws on an
     * unknown session, [ensureSession] confirms the session exists - creating
     * it locally if needed - *before* any socket is opened.
     */
    fun start(
        ctx: Context,
        sessionId: String,
        sources: List<String>,
        model: String,
        language: String,
        projectionIntent: Intent?,
    ) {
        if (running) stop()
        val wanted = sources.filter { it == SOURCE_MIC || it == SOURCE_DEVICE }.distinct()
        if (wanted.isEmpty()) return

        appContext = appContext ?: ctx.applicationContext
        val context = appContext ?: ctx

        // Create/confirm the session BEFORE opening sockets (markSocket is
        // called from the socket coroutines and requires a known session).
        val sid = ensureSession(context, sessionId, model, language)
        if (sid.isEmpty()) {
            Log.w(TAG, "could not create a live session; aborting capture")
            return
        }
        this.sessionId = sid
        startMs = System.currentTimeMillis()
        loss = false
        running = true
        this.sources.clear()
        this.sourceSkipReasons = emptyMap()

        // MediaProjection is only needed for device audio. The consent dialog
        // started by MediaProjectionManager.createScreenCaptureIntent() returns
        // Activity.RESULT_OK on success, which is the result code used here.
        var proj: MediaProjection? = null
        if (SOURCE_DEVICE in wanted && projectionIntent != null) {
            proj = try {
                val mgr = context.getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                        as MediaProjectionManager
                mgr.getMediaProjection(Activity.RESULT_OK, projectionIntent)
            } catch (e: Exception) {
                Log.w(TAG, "MediaProjection unavailable: ${e.message}")
                null
            }
        }
        projection = proj

        // Build one AudioRecord per requested source. Permission gating is done
        // by the Activity before start() is called.
        val skipReasons = LinkedHashMap<String, String>()
        for (source in wanted) {
            if (source == SOURCE_DEVICE && proj == null) {
                skipReasons[source] = "media projection consent denied or unavailable"
                continue
            }
            val record = buildRecord(source, proj)
            if (record == null) {
                skipReasons[source] = "audio record unavailable"
                continue
            }
            val st = SourceState(source, sid)
            st.record = record
            this.sources[source] = st
        }
        this.sourceSkipReasons = skipReasons
        if (this.sources.isEmpty()) {
            Log.w(TAG, "no usable audio source; nothing started")
            running = false
            proj?.stop()
            projection = null
            val ctx = appContext
            val sid = this.sessionId
            if (ctx != null && sid.isNotEmpty()) {
                try { LiveSessionStore.close(ctx, sid) } catch (_: Exception) { }
            }
            this.sessionId = ""
            return
        }

        // One OkHttp/Ktor WebSocket per source, plus one capture loop per source.
        for ((_, st) in this.sources) {
            openSocket(st, model, language)
            startCapture(st)
        }
    }

    /**
     * Stop capture, flush the final partial chunk, drain in-flight commits with
     * a bounded [awaitPending], then POST close. No PCM bytes are written.
     */
    fun stop() {
        if (!running && sessionId.isEmpty()) return
        running = false

        // 1. Freeze capture and flush the final (< 5 s) chunk, then pump any
        //    queued chunks into the socket outboxes.
        for (st in sources.values) {
            st.captureJob?.cancel()
            cutChunk(st, System.currentTimeMillis())
            pump(st)
        }

        // 2. Bounded drain: wait for every commit's completed/failed callback
        //    (and its append POST) before the files are closed.
        val drained = awaitPending(DRAIN_TIMEOUT_MS)
        if (!drained) {
            Log.w(TAG, "drain timed out; some transcripts may be missing")
        }

        // 3. Close the session on the local server.
        val sid = sessionId
        if (sid.isNotEmpty()) postClose(sid)

        // 4. Release records, sockets and the projection.
        for (st in sources.values) {
            try { st.record?.stop() } catch (_: Exception) {}
            try { st.record?.release() } catch (_: Exception) {}
            st.socketJob?.cancel()
        }
        try { projection?.stop() } catch (_: Exception) {}
        projection = null
        sources.clear()
        sessionId = ""
        withPending {
            outstanding = 0
            signalPending()
        }
    }

    /** Snapshot of controller state for the UI status poller. Pure JVM. */
    fun status(): Map<String, Any> {
        val pending = withPending { outstanding }
        return mapOf(
            "running" to running,
            "session_id" to sessionId,
            "sources" to sources.keys.toList().sorted(),
            "source_reasons" to sourceSkipReasons,
            "pending" to pending,
            "loss" to loss,
        )
    }

    /**
     * Wait up to [timeoutMs] for all outstanding (queued + in-flight) commits
     * to complete. Returns true when the queue is drained, false on timeout.
     * Pure JVM: it only uses a lock/condition wait, so it is testable without
     * any Android framework, socket or audio device.
     */
    fun awaitPending(timeoutMs: Long): Boolean {
        pendingLock.lock()
        try {
            val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs)
            while (outstanding > 0) {
                val remaining = deadline - System.nanoTime()
                if (remaining <= 0) return false
                try {
                    pendingChanged.awaitNanos(remaining)
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return false
                }
            }
            return true
        } finally {
            pendingLock.unlock()
        }
    }

    /** Test hook: inject a fake in-flight commit queue of [count] commits. */
    fun setPendingForTest(count: Int) {
        withPending {
            outstanding = count.coerceAtLeast(0)
            signalPending()
        }
    }

    /** Test hook: deliver one `completed` callback (drain one commit). */
    fun completePendingForTest() {
        withPending {
            if (outstanding > 0) outstanding--
            signalPending()
        }
    }

    // --------------------------------------------------------------- session / records

    /** Ensure the store knows [sessionId] before sockets open; create if absent. */
    private fun ensureSession(
        ctx: Context,
        sessionId: String,
        model: String,
        language: String,
    ): String = try {
        LiveSessionStore.status(sessionId) // throws NoSuchElementException if unknown
        sessionId
    } catch (e: NoSuchElementException) {
        @Suppress("UNCHECKED_CAST")
        (LiveSessionStore.create(ctx, language, model, "android")["id"] as? String) ?: ""
    }

    /**
     * Mic uses `AudioSource.MIC`. Device audio uses an
     * [AudioPlaybackCaptureConfiguration] built from the [MediaProjection] and
     * passed to `AudioRecord.Builder.setAudioPlaybackCaptureConfig` - this is
     * the real Android API (there is no `MediaProjection.createAudioRecord`,
     * and no `addMatchAllExcludedActivities` helper). Building the capture
     * config with no matching filters captures all playback; the system already
     * excludes apps that opt out of playback capture.
     */
    private fun buildRecord(source: String, proj: MediaProjection?): AudioRecord? {
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .setSampleRate(SAMPLE_RATE_HZ)
            .build()
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE_HZ, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)
        return try {
            val builder = AudioRecord.Builder()
                .setAudioFormat(format)
                .setBufferSizeInBytes(bufSize)
            if (source == SOURCE_MIC) {
                builder.setAudioSource(MediaRecorder.AudioSource.MIC)
            } else {
                val p = proj ?: return null
                val config = AudioPlaybackCaptureConfiguration.Builder(p).build()
                builder.setAudioPlaybackCaptureConfig(config)
            }
            val record = builder.build()
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                record.release()
                null
            } else {
                record
            }
        } catch (e: Exception) {
            Log.w(TAG, "AudioRecord($source) unavailable: ${e.message}")
            null
        }
    }

    // --------------------------------------------------------------- capture loop

    private fun startCapture(st: SourceState) {
        val record = st.record ?: return
        st.captureJob = ioScope.launch(Dispatchers.IO) {
            val buf = ByteArray(READ_SIZE)
            st.lastCutMs = System.currentTimeMillis()
            try {
                record.startRecording()
            } catch (e: Exception) {
                Log.w(TAG, "startRecording(${st.source}) failed: ${e.message}")
                return@launch
            }
            while (isActive && running) {
                val n = try {
                    record.read(buf, 0, buf.size)
                } catch (e: Exception) {
                    -1
                }
                if (n > 0) {
                    synchronized(st.buffer) { st.buffer.write(buf, 0, n) }
                } else if (n < 0) {
                    break
                }
                val now = System.currentTimeMillis()
                if (now - st.lastCutMs >= COMMIT_MS) cutChunk(st, now)
            }
        }
    }

    /** Emit the buffered PCM as one chunk (~5000 ms cadence, never blocking). */
    private fun cutChunk(st: SourceState, now: Long) {
        val pcm = synchronized(st.buffer) {
            val bytes = st.buffer.toByteArray()
            st.buffer.reset()
            bytes
        }
        st.lastCutMs = now
        if (pcm.isEmpty()) return
        enqueue(st, Chunk(pcm, (now - startMs).coerceAtLeast(0L)))
    }

    /** Enqueue a chunk; log and drop the oldest when the queue is full. */
    private fun enqueue(st: SourceState, chunk: Chunk) {
        synchronized(st.queue) {
            st.queue.addLast(chunk)
            withPending { outstanding++ }
            while (st.queue.size > MAX_QUEUE) {
                val dropped = st.queue.removeFirst()
                withPending {
                    if (outstanding > 0) outstanding--
                    signalPending()
                }
                loss = true
                val seconds = dropped.pcm.size / (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)
                Log.w(TAG, "source ${st.source} queue full, dropped oldest " +
                        "unprocessed ${seconds}s chunk")
            }
        }
        pump(st)
    }

    /** Move queued chunks into the socket outbox, up to MAX_INFLIGHT. */
    private fun pump(st: SourceState) {
        if (!st.socketOpen) return
        while (true) {
            var next: Chunk? = null
            synchronized(st.queue) {
                if (st.inFlight < MAX_INFLIGHT && st.queue.isNotEmpty()) {
                    next = st.queue.removeFirst()
                    st.inFlight++
                }
            }
            val chunk = next ?: break
            st.outbox.trySend(chunk)
        }
    }

    // --------------------------------------------------------------- WebSocket

    private fun openSocket(st: SourceState, model: String, language: String) {
        st.socketJob = ioScope.launch(Dispatchers.IO) {
            var marked = false
            try {
                try {
                    LiveSessionStore.markSocket(st.sid, +1)
                    marked = true
                } catch (e: Exception) {
                    Log.w(TAG, "markSocket open failed: ${e.message}")
                }
                wsClient.webSocket(WS_URL) {
                    val session = this
                    // Announce 16 kHz mono PCM16 and the transcription model.
                    session.send(Frame.Text(sessionUpdateJson(model, language)))
                    st.socketOpen = true
                    // A single sender serialises append+commit pairs per source.
                    val sender = launch {
                        for (chunk in st.outbox) {
                            val b64 = Base64.encodeToString(chunk.pcm, Base64.NO_WRAP)
                            session.send(Frame.Text(appendJson(b64)))
                            session.send(Frame.Text(commitJson()))
                            synchronized(st.awaitingElapsed) {
                                st.awaitingElapsed.addLast(chunk.elapsedMs)
                            }
                        }
                    }
                    try {
                        pump(st) // flush chunks buffered while connecting
                        for (frame in incoming) {
                            if (frame is Frame.Text) handleFrame(st, frame.readText())
                        }
                    } finally {
                        sender.cancel()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "socket ${st.source}: ${e.message}")
            } finally {
                st.socketOpen = false
                if (marked) {
                    try {
                        LiveSessionStore.markSocket(st.sid, -1)
                    } catch (e: Exception) {
                        Log.w(TAG, "markSocket close failed: ${e.message}")
                    }
                }
            }
        }
    }

    private fun handleFrame(st: SourceState, text: String) {
        val msg = try {
            JsonParser.parseString(text).asJsonObject
        } catch (_: Exception) {
            return
        }
        when (msg.get("type")?.asString) {
            "input_audio_buffer.committed" -> {
                val id = msg.get("item_id")?.asString ?: return
                val elapsed = synchronized(st.awaitingElapsed) {
                    st.awaitingElapsed.pollFirst()
                } ?: elapsedNow()
                synchronized(st.itemElapsed) { st.itemElapsed[id] = elapsed }
            }
            "conversation.item.input_audio_transcription.completed" -> {
                val id = msg.get("item_id")?.asString ?: ""
                val transcript = msg.get("transcript")?.asString ?: ""
                val elapsed = synchronized(st.itemElapsed) {
                    st.itemElapsed.remove(id)
                } ?: elapsedNow()
                // Dedupe is server-side on item_id; text comes from the event.
                postAppend(st.sid, st.source, transcript, id, elapsed)
                onCompleted(st)
            }
            "conversation.item.input_audio_transcription.failed" -> {
                Log.w(TAG, "transcription failed for ${st.source}")
                onCompleted(st)
            }
        }
    }

    /** One commit's completed/failed callback arrived: free a slot, keep pumping. */
    private fun onCompleted(st: SourceState) {
        synchronized(st.queue) { if (st.inFlight > 0) st.inFlight-- }
        withPending {
            if (outstanding > 0) outstanding--
            signalPending()
        }
        pump(st)
    }

    // --------------------------------------------------------------- messages / HTTP

    private fun sessionUpdateJson(model: String, language: String): String =
        JsonObject().apply {
            addProperty("type", "session.update")
            add("session", JsonObject().apply {
                addProperty("type", "transcription")
                add("audio", JsonObject().apply {
                    add("input", JsonObject().apply {
                        add("format", JsonObject().apply {
                            addProperty("type", "audio/pcm")
                            addProperty("rate", SAMPLE_RATE_HZ)
                        })
                        add("transcription", JsonObject().apply {
                            addProperty("model", model)
                            if (language.isNotBlank()) addProperty("language", language)
                        })
                    })
                })
            })
        }.toString()

    private fun appendJson(base64Pcm: String): String = JsonObject().apply {
        addProperty("type", "input_audio_buffer.append")
        addProperty("audio", base64Pcm)
    }.toString()

    private fun commitJson(): String = JsonObject().apply {
        addProperty("type", "input_audio_buffer.commit")
    }.toString()

    private fun elapsedNow(): Long = (System.currentTimeMillis() - startMs).coerceAtLeast(0L)

    /** POST an append; only transcript text is sent, never audio bytes. */
    private fun postAppend(sid: String, source: String, text: String, itemId: String, elapsedMs: Long) {
        val body = JsonObject().apply {
            addProperty("source", source)
            addProperty("text", text)
            addProperty("item_id", itemId)
            addProperty("elapsed_ms", elapsedMs.coerceAtLeast(0L))
        }.toString()
        postJson("/v1/live/sessions/$sid/append", body)
    }

    private fun postClose(sid: String) {
        postJson("/v1/live/sessions/$sid/close", "{}")
    }

    /** Synchronous localhost POST (OkHttp); failures are logged, never fatal. */
    private fun postJson(path: String, json: String) {
        try {
            val req = Request.Builder()
                .url("$HTTP_BASE$path")
                .post(json.toRequestBody("application/json".toMediaType()))
                .build()
            postClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    Log.w(TAG, "POST $path -> HTTP ${resp.code}")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "POST $path failed: ${e.message}")
        }
    }
}
