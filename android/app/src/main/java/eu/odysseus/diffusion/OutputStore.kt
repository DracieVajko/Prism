package eu.odysseus.diffusion

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Data class representing an output file on the device. */
data class OutputFile(
    val name: String,
    val kind: String,       // image, video, audio, text
    val size: Long,
    val path: String,       // path within output/ folder
    val uri: android.net.Uri, // content:// URI via FileProvider
    val lastModified: Long,
)

/**
 * Default output folder for every job on the phone.
 *
 * Files live under `Android/data/<pkg>/files/output/<kind>/` so the server can
 * list and serve them, and "Save to Downloads" copies them into the public
 * `Download/Prism` folder where any file manager can open them.
 */
object OutputStore {
    private const val TAG = "OutputStore"
    private val KINDS = listOf("image", "video", "audio", "text")

    fun root(ctx: Context): File =
        File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "output").apply { mkdirs() }

    fun kindDir(ctx: Context, kind: String): File = File(root(ctx), kind).apply { mkdirs() }

    /** Write one artifact; returns the descriptor used in API responses. */
    fun store(ctx: Context, kind: String, ext: String, data: ByteArray, rid: String = ""): Map<String, Any> {
        val safeKind = if (kind in KINDS) kind else "text"
        val safeExt = if (ext.startsWith(".")) ext else ".$ext"
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val safeRid = rid.filter { it.isLetterOrDigit() || it == '-' || it == '_' }.take(24).trim('-', '_')
        val base = "$stamp-${safeRid.ifEmpty { System.currentTimeMillis().toString(36) }}"
        var target = File(kindDir(ctx, safeKind), "$base$safeExt")
        var n = 1
        while (target.exists()) {
            target = File(kindDir(ctx, safeKind), "$base-$n$safeExt")
            n++
        }
        target.writeBytes(data)
        return mapOf(
            "name" to target.name,
            "kind" to safeKind,
            "size" to data.size,
            "path" to "$safeKind/${target.name}",
            "url" to "/v1/outputs/$safeKind/${target.name}",
        )
    }

    /**
     * List output files, optionally filtered by kind and sorted by the given criterion.
     *
     * @param ctx Context
     * @param kind Optional filter: "image", "video", "audio", "text", or null for all
     * @param limit Max number of files to return
     * @param sortBy One of: "date", "name", "size"
     * @return List of OutputFile objects sorted accordingly
     */
    fun list(ctx: Context, kind: String? = null, limit: Int = 200, sortBy: String = "date"): List<OutputFile> {
        val kinds = if (kind != null && kind in KINDS) listOf(kind) else KINDS
        val all = ArrayList<OutputFile>()

        for (k in kinds) {
            val dir = kindDir(ctx, k)
            dir.listFiles()?.forEach { f ->
                if (f.isFile) {
                    val mime = mimeOf(f)
                    val uri = android.net.Uri.fromFile(f)
                    all.add(OutputFile(
                        name = f.name,
                        kind = k,
                        size = f.length(),
                        path = "$k/${f.name}",
                        uri = uri,
                        lastModified = f.lastModified(),
                    ))
                }
            }
        }

        return when (sortBy) {
            "name" -> all.sortedBy { it.name.lowercase() }
            "size" -> all.sortedBy { it.size }
            else -> all.sortedByDescending { it.lastModified } // date
        }.take(limit)
    }

    /** Total count and total bytes across all kinds (or a specific kind). */
    fun total(ctx: Context, kind: String? = null): Pair<Int, Long> {
        val files = list(ctx, kind, Int.MAX_VALUE, "date")
        return files.size to files.sumOf { it.size }
    }

    /** Copy every output (or only those of a given kind) into the public Download/Prism folder via MediaStore. */
    fun saveAllToDownloads(ctx: Context, kind: String? = null): Int {
        var saved = 0
        for (f in list(ctx, kind, 1000, "date")) {
            val outFile = File(kindDir(ctx, f.kind), f.name)
            if (!outFile.exists()) continue
            try {
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, f.name)
                    put(MediaStore.MediaColumns.MIME_TYPE, mimeOf(outFile))
                    put(MediaStore.MediaColumns.RELATIVE_PATH,
                        "${android.os.Environment.DIRECTORY_DOWNLOADS}/Prism")
                }
                val uri: Uri? = ctx.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    ctx.contentResolver.openOutputStream(uri)?.use { out ->
                        outFile.inputStream().use { it.copyTo(out) }
                    }
                    saved++
                }
            } catch (e: Exception) {
                Log.w(TAG, "save ${f.name} failed: ${e.message}")
            }
        }
        return saved
    }

    /** Delete a single output file. */
    fun delete(file: File): Boolean = runCatching { file.delete() }.getOrDefault(false)

    /** Delete all output files (optionally filtered by kind). */
    fun deleteAll(ctx: Context, kind: String? = null): Int {
        var deleted = 0
        val kinds = if (kind != null && kind in KINDS) listOf(kind) else KINDS
        for (k in kinds) {
            kindDir(ctx, k).listFiles()?.forEach { f ->
                if (f.isFile && f.delete()) deleted++
            }
        }
        return deleted
    }

    /** Delete a single file by its path within output/ (e.g. "image/foo.png"). */
    fun deleteByPath(ctx: Context, relPath: String): Boolean {
        val parts = relPath.split("/", limit = 2)
        if (parts.size < 2) return false
        val kind = parts[0]
        val name = parts[1]
        val f = File(kindDir(ctx, kind), name)
        return f.delete()
    }

    private fun mimeOf(f: File): String = when (f.extension.lowercase()) {
        "png" -> "image/png"
        "jpg", "jpeg" -> "image/jpeg"
        "webp" -> "image/webp"
        "mp4" -> "video/mp4"
        "m4a" -> "audio/mp4"
        "mp3" -> "audio/mpeg"
        "wav" -> "audio/wav"
        "txt" -> "text/plain"
        else -> "application/octet-stream"
    }
}
