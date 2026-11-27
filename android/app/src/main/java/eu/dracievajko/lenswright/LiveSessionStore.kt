package eu.dracievajko.lenswright

import android.content.Context
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * In-memory registry of live transcription sessions backed by two append-only
 * text files per session (one for "mic", one for "device").
 *
 * This mirrors the PC LiveSessionStore semantics: a header line on create,
 * `[mm:ss] text` lines flushed on every append, idempotent `item_id` dedupe,
 * and a 30-minute idle sweep that only closes sessions with no open sockets.
 * Only transcript text is ever written; no audio bytes reach disk.
 *
 * Exceptions mirror the PC store and are mapped to HTTP codes by ApiServer:
 * - [NoSuchElementException] unknown session -> 404
 * - [IllegalArgumentException] bad source -> 400
 * - [IllegalStateException] session closed -> 409
 */
object LiveSessionStore {
    /** The only two transcript sources. */
    val SOURCES: List<String> = listOf("mic", "device")

    private const val DEFAULT_MAX_IDLE_SECONDS = 1800

    private val sessions = ConcurrentHashMap<String, LiveSession>()
    // Reentrant monitor, mirroring the PC store's RLock: every public method
    // runs under it so append/socket/close/sweep are atomic with respect to
    // each other. close() is called from sweepIdle() under the same lock.
    private val lock = Any()

    private class LiveSession(
        val id: String,
        val language: String,
        val model: String,
        val label: String,
        val created: Double,
    ) {
        val paths = HashMap<String, File>()
        val writers = HashMap<String, BufferedWriter>()
        val lines = HashMap<String, Int>()
        val seen = HashMap<String, MutableSet<String>>()
        var closed: Boolean = false
        var openSockets: Int = 0
        var lastActivity: Double = created
    }

    /** Current wall-clock time in Unix seconds (matches the PC store's time.time()). */
    private fun nowSeconds(): Double = System.currentTimeMillis() / 1000.0

    private fun get(sessionId: String): LiveSession =
        sessions[sessionId] ?: throw NoSuchElementException("unknown live session: $sessionId")

    private fun checkSource(source: String) {
        if (source !in SOURCES) {
            throw IllegalArgumentException("unknown live source: $source")
        }
    }

    /** Format elapsed milliseconds as the `[mm:ss]` transcript prefix. */
    private fun stampPrefix(elapsedMs: Long): String {
        val totalS = elapsedMs.coerceAtLeast(0L) / 1000L
        return String.format(Locale.US, "[%02d:%02d]", totalS / 60L, totalS % 60L)
    }

    /** Public descriptor: relative path, matching OutputStore.store() and the PC store. */
    private fun descriptor(file: File): Map<String, Any> = mapOf(
        "name" to file.name,
        "kind" to "text",
        "size" to file.length(),
        "path" to "text/${file.name}",
        "url" to "/v1/outputs/text/${file.name}",
    )

    private fun descriptorsOf(session: LiveSession): Map<String, Any> =
        SOURCES.associateWith { descriptor(session.paths[it]!!) }

    /** Open two transcript files and return the new session summary. */
    fun create(
        ctx: Context,
        language: String = "",
        model: String = "turbo",
        label: String = "",
    ): Map<String, Any> = synchronized(lock) {
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val sid = "live_${stamp}_${UUID.randomUUID().toString().replace("-", "").take(6)}"
        val created = nowSeconds()
        // A header must stay a single line even if a caller passes a language
        // string containing a newline.
        val headerLanguage = language.replace("\r", " ").replace("\n", " ")
        val session = LiveSession(
            id = sid,
            language = language,
            model = model,
            label = label,
            created = created,
        )
        try {
            for (source in SOURCES) {
                val file = OutputStore.liveFile(ctx, sid, source)
                val writer = BufferedWriter(
                    OutputStreamWriter(FileOutputStream(file, true), Charsets.UTF_8))
                writer.write(
                    "# Live transcript - source: $source - language: $headerLanguage\n")
                writer.flush()
                session.paths[source] = file
                session.writers[source] = writer
                session.lines[source] = 1
                session.seen[source] = HashSet()
            }
        } catch (e: Exception) {
            // Do not leak a writer if the second file could not be opened.
            session.writers.values.forEach { runCatching { it.close() } }
            throw e
        }
        sessions[sid] = session
        mapOf(
            "id" to session.id,
            "sources" to SOURCES,
            "created" to session.created,
            "language" to session.language,
            "model" to session.model,
            "label" to session.label,
            "closed" to false,
            "descriptors" to descriptorsOf(session),
        )
    }

    /** Append one timestamped line, ignoring empty text and repeated ids. */
    fun append(
        sessionId: String,
        source: String,
        text: String,
        itemId: String = "",
        elapsedMs: Long = 0,
    ): Map<String, Any> = synchronized(lock) {
        val session = get(sessionId)
        if (session.closed) {
            throw IllegalStateException("live session is closed: $sessionId")
        }
        checkSource(source)
        session.lastActivity = nowSeconds()
        val seen = session.seen[source]!!
        val currentLine = session.lines[source]!!
        // One utterance must occupy exactly one line in the transcript.
        val fragment = text
            .replace("\r\n", " ").replace("\r", " ").replace("\n", " ").trim()
        when {
            itemId.isNotEmpty() && itemId in seen -> mapOf(
                "ok" to true, "duplicate" to true, "bytes" to 0,
                "line" to currentLine,
            )
            fragment.isEmpty() -> mapOf(
                "ok" to true, "duplicate" to false, "bytes" to 0,
                "line" to currentLine,
            )
            else -> {
                val line = "${stampPrefix(elapsedMs)} $fragment\n"
                val writer = session.writers[source]!!
                writer.write(line)
                writer.flush()
                // Only a written line consumes the id; an empty append leaves
                // the id free so a retransmit with real text is still recorded.
                if (itemId.isNotEmpty()) seen.add(itemId)
                val nextLine = currentLine + 1
                session.lines[source] = nextLine
                mapOf(
                    "ok" to true,
                    "duplicate" to false,
                    "bytes" to line.toByteArray(Charsets.UTF_8).size,
                    "line" to nextLine,
                )
            }
        }
    }

    /**
     * Flush and close both files, mirror them to Downloads once, and mark the
     * session closed. Repeated calls are harmless.
     */
    fun close(ctx: Context, sessionId: String): Map<String, Any> = synchronized(lock) {
        val session = get(sessionId)
        if (!session.closed) {
            session.writers.values.forEach { w ->
                runCatching { w.flush() }
                runCatching { w.close() }
            }
            session.writers.clear()
            for (source in SOURCES) {
                session.paths[source]?.let { OutputStore.publish(ctx, it) }
            }
            session.closed = true
        }
        mapOf(
            "ok" to true,
            "closed" to true,
            "descriptors" to descriptorsOf(session),
            // Flat keys kept for parity with the PC store's close() shape.
            "mic" to descriptor(session.paths["mic"]!!),
            "device" to descriptor(session.paths["device"]!!),
            "lines" to HashMap(session.lines),
        )
    }

    /** Return session metadata, descriptors and per-source line counts. */
    fun status(sessionId: String): Map<String, Any> = synchronized(lock) {
        val session = get(sessionId)
        mapOf(
            "id" to session.id,
            "language" to session.language,
            "model" to session.model,
            "label" to session.label,
            "created" to session.created,
            "sources" to SOURCES,
            "descriptors" to descriptorsOf(session),
            "lines" to HashMap(session.lines),
            "closed" to session.closed,
            "open_sockets" to session.openSockets,
            "last_activity" to session.lastActivity,
        )
    }

    /** Record a socket opening (+1) or closing (-1) for a session. */
    fun markSocket(sessionId: String, delta: Int) {
        synchronized(lock) {
            val session = get(sessionId)
            session.openSockets = maxOf(0, session.openSockets + delta)
            // An opening counts as activity; a closing does not reset the idle
            // clock, so an abandoned session becomes sweepable promptly.
            if (delta > 0) session.lastActivity = nowSeconds()
        }
    }

    /**
     * Test/support hook: force the idle timestamp to `ts` (Unix seconds).
     * Accepts any numeric literal, mirroring the PC store's dynamic helper.
     */
    fun lastActivity(sessionId: String, ts: Number) {
        synchronized(lock) {
            get(sessionId).lastActivity = ts.toDouble()
        }
    }

    /** Close sessions idle longer than `maxIdleSeconds` with no open sockets. */
    fun sweepIdle(ctx: Context, maxIdleSeconds: Int = DEFAULT_MAX_IDLE_SECONDS): Int =
        synchronized(lock) {
            val now = nowSeconds()
            var swept = 0
            for (session in ArrayList(sessions.values)) {
                if (session.closed || session.openSockets > 0) continue
                if (now - session.lastActivity < maxIdleSeconds.toDouble()) continue
                close(ctx, session.id)
                swept++
            }
            swept
        }
}
