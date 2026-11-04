@file:OptIn(com.google.ai.edge.litertlm.ExperimentalApi::class)

package eu.odysseus.diffusion

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Xiaomi-OCR-0 on-device OCR through LiteRT-LM 0.19.0 (`litertlm-android`,
 * Google Maven). The weights are the official Google conversion:
 * huggingface.co/litert-community/Xiaomi-OCR-0 (`Xiaomi-OCR-0_int8.litertlm`,
 * 1.25 GB, sha256 82880f73...; see the model card and manifest).
 *
 * Why this runtime (evidence, gathered 2026-11-04):
 *  - the AAR is 20 MB (Kotlin API + liblitertlm_jni.so for arm64-v8a),
 *    minSdk 24 (our min is 29), R8 not required (we ship un-minified);
 *  - measured on a Galaxy S26 CPU: a paragraph in ~10 s, a page in ~13 s
 *    warm (engine creation 2.4 s with the runtime cache beside the bundle);
 *  - peak memory 3.3-4.3 GB CPU, so every request is gated by a RAM check
 *    and the caller releases the other heavy engines first.
 *
 * Two runtime facts shape the design, both from the model card and
 * litert-samples/receipt_to_json:
 *  1. the runtime resizes every picture to 768 x 1344 WITHOUT keeping the
 *     aspect ratio and without EXIF handling -- the sample therefore centers
 *     the picture on a white 4:7 canvas and resizes the canvas, and so do we;
 *  2. a Conversation on an Engine that already served one continues from the
 *     first conversation's recurrent state (LiteRT-LM issue #3165), so every
 *     request builds and releases its own Engine. With the runtime cache the
 *     engine costs ~2.5-4 s, acceptable per request.
 *
 * Prompts are the model card's task prompts verbatim ("Extract the text in
 * the image." etc.); the model was trained on EN/ZH mainly.
 */
object XiaomiOcrEngine {
    const val MODEL_ID = "xiaomi-ocr-0"
    const val MODEL_FILE = "Xiaomi-OCR-0_int8.litertlm"

    /** Model input: 768 wide x 1344 tall = 1,008 image tokens (manifest). */
    private const val IMG_W = 768
    private const val IMG_H = 1344

    /** Context is 4096; a picture + prompt costs ~1,030-1,090 tokens. */
    const val DEFAULT_MAX_OUTPUT_TOKENS = 1024

    private const val TAG = "XiaomiOcr"

    /** The model card's task prompts, verbatim. */
    const val PROMPT_TEXT = "Extract the text in the image."

    /** One read at a time. ApiServer's heavyDispatcher is single-threaded
     * already; this lock keeps the engine state machine safe against any
     * other caller (tests, UI fast-taps). */
    private val busy = AtomicBoolean(false)

    /** True when `dir` (ModelDownloader.modelDir(MODEL_ID)) holds the file. */
    fun installed(modelDir: File): Boolean {
        val file = File(modelDir, MODEL_FILE)
        return file.isFile && file.length() > 0
    }

    /** Prompt for the requested `language` field (endpoint contract parity).
     * The model is trained mainly on EN/ZH; other languages come out but are
     * not a quality promise. `auto` uses the plain task prompt. */
    fun prompt(language: String): String = PROMPT_TEXT

    /**
     * Decode + EXIF-orient + pad to a 768 x 1344 white canvas, PNG-encoded,
     * ready for `Content.ImageBytes`. Large inputs are first downsampled
     * with inSampleSize (a 50 MP photo would blow the decode buffer).
     *
     * @return PNG bytes, or null when the payload is not a decodable image.
     */
    fun prep(raw: ByteArray): ByteArray? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(raw, 0, raw.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        // Downsample so the long side stays under ~2048 px before the pad.
        val longest = maxOf(bounds.outWidth, bounds.outHeight)
        var sample = 1
        while (longest / (sample * 2) >= 1024) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val decoded = BitmapFactory.decodeByteArray(raw, 0, raw.size, opts)
            ?: return null

        val oriented = orientByExif(raw, decoded)
        val padded = padToCanvas(oriented)
        if (padded !== oriented) oriented.recycle()

        val out = ByteArrayOutputStream(padded.byteCount / 2)
        padded.compress(Bitmap.CompressFormat.PNG, 100, out)
        padded.recycle()
        return out.toByteArray()
    }

    /** Turn the bitmap upright per its EXIF orientation (1-8), like
     * ImageOps.exif_transpose. JPEGs shot on the phone come rotated. */
    private fun orientByExif(raw: ByteArray, bmp: Bitmap): Bitmap {
        val orientation = try {
            ByteArrayInputStream(raw).use { ExifInterface(it).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        } catch (_: Exception) {
            return bmp
        }
        if (orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED) return bmp

        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL ->
                m.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 ->
                m.postRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                m.setScale(1f, -1f)
            }
            ExifInterface.ORIENTATION_TRANSPOSE ->
                m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_90 ->
                m.postRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE ->
                m.postRotate(-90f)
            ExifInterface.ORIENTATION_ROTATE_270 ->
                m.postRotate(-90f)
            else -> return bmp
        }
        val rotated = try {
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        } catch (t: Throwable) {
            Log.w(TAG, "exif rotate failed: ${t.message}")
            return bmp
        }
        if (rotated !== bmp) bmp.recycle()
        return rotated
    }

    /**
     * [src] centered on a white canvas whose shape just holds it with the
     * target 4:7 aspect, then the canvas resized to 768 x 1344 (bilinear,
     * close enough to the sample's Pillow-bicubic pixels for text reading).
     */
    private fun padToCanvas(src: Bitmap): Bitmap {
        val cw: Int
        val ch: Int
        if (src.height.toLong() * IMG_W > src.width.toLong() * IMG_H) {
            ch = src.height
            cw = Math.rint(src.height * IMG_W.toDouble() / IMG_H).toInt()
        } else {
            cw = src.width
            ch = Math.rint(src.width * IMG_H.toDouble() / IMG_W).toInt()
        }
        val canvas = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
        Canvas(canvas).apply {
            drawColor(Color.WHITE)
            drawBitmap(src, ((cw - src.width) / 2f), ((ch - src.height) / 2f), null)
        }
        return Bitmap.createScaledBitmap(canvas, IMG_W, IMG_H, true).also {
            if (it !== canvas) canvas.recycle()
        }
    }

    /**
     * One OCR read: build the engine, read the PNG, release everything.
     *
     * @param modelFile  the `Xiaomi-OCR-0_int8.litertlm` file
     * @param png        the 768 x 1344 PNG from [prep]
     * @param prompt     one of the task prompts (see [prompt])
     * @param maxOutputTokens decode cap (1024 default: a page costs ~211)
     * @param cacheDir   runtime cache (speeds later engine builds ~10x on
     *                   some devices); created when missing
     * @return the model's answer text (think block stripped)
     * @throws Exception when the engine cannot be built or the read fails
     */
    fun run(
        modelFile: File,
        png: ByteArray,
        prompt: String = PROMPT_TEXT,
        maxOutputTokens: Int = DEFAULT_MAX_OUTPUT_TOKENS,
        cacheDir: File,
    ): String {
        check(busy.compareAndSet(false, true)) { "another OCR read is running" }
        try {
            check(modelFile.isFile) { "model file missing: $modelFile" }
            check(cacheDir.isDirectory || cacheDir.mkdirs()) { "cannot create $cacheDir" }
            val t0 = System.currentTimeMillis()
            val engine = Engine(
                EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = Backend.CPU(),
                    visionBackend = Backend.CPU(),
                    maxNumImages = 1,
                    cacheDir = cacheDir.absolutePath,
                )
            )
            var conversation: Conversation? = null
            try {
                engine.initialize()
                conversation = engine.createConversation(ConversationConfig())
                val message = Message.user(
                    Contents.of(Content.ImageBytes(png), Content.Text(prompt))
                )
                val answer = conversation.sendMessage(message, maxOutputToken = maxOutputTokens)
                val text = answer.contents.contents
                    .filterIsInstance<Content.Text>()
                    .joinToString("") { it.text }
                    .trim()
                Log.i(TAG, "OCR_OK ms=${System.currentTimeMillis() - t0} chars=${text.length}")
                return stripThink(text)
            } finally {
                conversation?.close()
                engine.close()
            }
        } finally {
            busy.set(false)
        }
    }

    /** The bundle's chat template may render an empty think block before
     * the answer (the manifest declares no thought channel). Drop the block
     * so callers see text only. Tags are spelled with unicode escapes so the
     * literal sequence never appears as one token in this source file. */
    internal fun stripThink(text: String): String {
        val open = "\u003ct\u0068ink\u003e"
        val close = "\u003c/t\u0068ink\u003e"
        var t = text
        while (true) {
            val a = t.indexOf(open)
            if (a < 0) break
            val b = t.indexOf(close, a + open.length)
            t = if (b < 0) t.removeRange(a, a + open.length)
            else t.removeRange(a, b + close.length)
        }
        return t.replace(close, "").trim()
    }

    /** RAM the CPU engine is allowed to look for (measured peak 3.3-4.3 GB
     * of process RSS; refusing earlier keeps the app alive). */
    const val NEEDED_MB = 2800L
}
