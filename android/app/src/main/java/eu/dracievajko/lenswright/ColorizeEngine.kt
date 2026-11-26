package eu.dracievajko.lenswright

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * On-device photo colorization.
 *
 * Two model families are supported and both run through ONNX Runtime:
 *
 *  - `ddcolor-*`  predicts Lab chroma (a/b) at 512x512. The luminance of the
 *    source is kept, so the structure never changes. A global colour cast is
 *    removed by subtracting the per-image median of each a/b channel, and a
 *    guided filter keyed on the luminance cleans per-pixel chroma noise (that
 *    noise is what produced red/blue fringing on the PC build).
 *  - `deoldify-artistic` takes grayscale RGB in 0..255 and returns RGB in
 *    0..255. Its chroma is merged into the ORIGINAL luminance so the result is
 *    a colourised photo, not a redraw. This is the default, matching the PC.
 */
object ColorizeEngine {
    private const val TAG = "ColorizeEngine"
    const val DEFAULT_MODEL = "deoldify-artistic"

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var modelId: String = ""
    private var modelDir: File? = null
    private var modelType: String = ""
    private var root: File? = null

    @Volatile var lastError: String = ""

    private val MODEL_DIRS = mapOf(
        "deoldify-artistic" to "deoldify-artistic",
        "ddcolor-full" to "ddcolor-full",
        "ddcolor-artistic" to "ddcolor-artistic",
    )

    fun isInstalled(ctx: Context): Boolean = installedIds(ctx).isNotEmpty()

    fun installedIds(ctx: Context): List<String> = MODEL_DIRS.keys.filter { id ->
        val f = File(modelsRoot(ctx), MODEL_DIRS[id])
        f.isDirectory && f.walkTopDown().any { it.isFile && it.extension.equals("onnx", true) }
    }

    private fun modelsRoot(ctx: Context): File =
        (root ?: File(ctx.getExternalFilesDir(null), "models").also { root = it })

    private fun findOnnx(dir: File): File? =
        dir.walkTopDown().firstOrNull { it.isFile && it.extension.equals("onnx", true) }

    /** Load (or reuse) the ONNX session for [id]. */
    fun init(ctx: Context, id: String = DEFAULT_MODEL): Boolean {
        val resolved = if (MODEL_DIRS.containsKey(id)) id else DEFAULT_MODEL
        if (session != null && modelId == resolved) return true
        release()
        root = File(ctx.getExternalFilesDir(null), "models")
        val dir = File(modelsRoot(ctx), MODEL_DIRS[resolved] ?: resolved)
        val file = findOnnx(dir)
        if (file == null) {
            lastError = "no .onnx in ${dir.absolutePath}"
            Log.w(TAG, lastError)
            return false
        }
        return try {
            env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                val threads = HardwareProfile.optimalThreads(ctx)
                try { setIntraOpNumThreads(threads) } catch (_: Exception) { }
                try { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT) } catch (_: Exception) { }
            }
            session = (env as OrtEnvironment).createSession(file.absolutePath, opts)
            modelDir = dir
            modelId = resolved
            modelType = if (resolved.startsWith("ddcolor")) "ddcolor" else "deoldify"
            lastError = ""
            Log.i(TAG, "loaded $resolved from ${file.name}")
            true
        } catch (e: Throwable) {
            lastError = "load failed: ${e.message}"
            Log.e(TAG, lastError, e)
            release()
            false
        }
    }

    fun release(): Boolean {
        val had = session != null
        try { session?.close() } catch (_: Throwable) { }
        session = null
        modelDir = null
        modelId = ""
        modelType = ""
        return had
    }

    fun currentModel(): String = modelId

    /**
     * Colorize [src]. Returns a new bitmap at the source resolution, or null.
     */
    fun colorize(ctx: Context, src: Bitmap, id: String = DEFAULT_MODEL): Bitmap? {
        if (!init(ctx, id)) return null
        val s = session ?: return null
        return try {
            if (modelType == "ddcolor") colorizeDdcolor(s, src) else colorizeDeoldify(s, src)
        } catch (e: Throwable) {
            lastError = "colorize failed: ${e.message}"
            Log.e(TAG, lastError, e)
            null
        }
    }

    // ---------------------------------------------------------------- ddcolor

    private fun colorizeDdcolor(s: OrtSession, src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val gray = toGrayscale(src)

        // Model input: grey RGB in 0..1 at 512x512.
        val small = Bitmap.createScaledBitmap(gray, 512, 512, true)
        val px = IntArray(512 * 512)
        small.getPixels(px, 0, 512, 0, 0, 512, 512)
        val chw = FloatArray(3 * 512 * 512)
        val lGuide = FloatArray(512 * 512)
        for (i in px.indices) {
            val v = (px[i] and 0xFF) / 255f
            chw[i] = v
            chw[512 * 512 + i] = v
            chw[2 * 512 * 512 + i] = v
            lGuide[i] = lStar(v)
        }

        val inputName = s.inputNames.first()
        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(chw), longArrayOf(1, 3, 512, 512)
        )
        val ab = tensor.use { t ->
            s.run(mapOf(inputName to t)).use { out ->
                val raw = out.first().value.getValue()
                val arr = FloatArray(2 * 512 * 512)
                flattenFloats(raw, arr, 0)
                arr
            }
        }

        val a = FloatArray(512 * 512)
        val b = FloatArray(512 * 512)
        for (i in 0 until 512 * 512) {
            a[i] = ab[i]
            b[i] = ab[512 * 512 + i]
        }
        // Remove the checkpoint's global cast (median shift).
        recentre(a)
        recentre(b)
        // Edge-aware smoothing: kills the per-pixel chroma noise behind the
        // red/blue fringe. Same idea as the PC guided filter.
        val aS = GuidedFilter.smooth(lGuide, a, 512, 512, 6, 120f)
        val bS = GuidedFilter.smooth(lGuide, b, 512, 512, 6, 120f)

        val gain = if (modelId == "ddcolor-full") 3.0f else 2.2f
        val aUp = resizePlane(aS, 512, 512, w, h, gain)
        val bUp = resizePlane(bS, 512, 512, w, h, gain)

        // Luminance at full resolution from the (grayscale) source.
        val srcPx = IntArray(w * h)
        src.getPixels(srcPx, 0, w, 0, 0, w, h)
        val outPx = IntArray(w * h)
        for (i in srcPx.indices) {
            val v = (srcPx[i] and 0xFF) / 255f
            val L = lStar(v)
            val rgb = labToRgb(L, aUp[i], bUp[i])
            outPx[i] = rgb
        }
        return Bitmap.createBitmap(outPx, w, h, Bitmap.Config.ARGB_8888)
    }

    // --------------------------------------------------------------- deoldify

    private fun colorizeDeoldify(s: OrtSession, src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val gray = toGrayscale(src)
        val small = Bitmap.createScaledBitmap(gray, 256, 256, true)
        val px = IntArray(256 * 256)
        small.getPixels(px, 0, 256, 0, 0, 256, 256)

        // DeOldify contract: grayscale RGB scaled 0..255 (not 0..1).
        val chw = FloatArray(3 * 256 * 256)
        for (i in px.indices) {
            val v = (px[i] and 0xFF).toFloat()
            chw[i] = v
            chw[256 * 256 + i] = v
            chw[2 * 256 * 256 + i] = v
        }
        val inputName = s.inputNames.first()
        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(chw), longArrayOf(1, 3, 256, 256)
        )
        val rgb = tensor.use { t ->
            s.run(mapOf(inputName to t)).use { out ->
                val raw = out.first().value.getValue()
                val arr = FloatArray(3 * 256 * 256)
                flattenFloats(raw, arr, 0)
                arr
            }
        }

        // Take only the chroma (a/b) from the model and keep the source
        // luminance, so faces and structure stay the same.
        val size = 256
        val a = FloatArray(size * size)
        val b = FloatArray(size * size)
        val srcPx = IntArray(w * h)
        src.getPixels(srcPx, 0, w, 0, 0, w, h)
        for (i in 0 until size * size) {
            val r = rgb[i].coerceIn(0f, 255f)
            val g = rgb[size * size + i].coerceIn(0f, 255f)
            val bl = rgb[2 * size * size + i].coerceIn(0f, 255f)
            val (_, aa, bb) = rgbToLab(r / 255f, g / 255f, bl / 255f)
            a[i] = aa
            b[i] = bb
        }
        val aUp = resizePlane(a, size, size, w, h, 1f)
        val bUp = resizePlane(b, size, size, w, h, 1f)
        val outPx = IntArray(w * h)
        for (i in srcPx.indices) {
            val v = (srcPx[i] and 0xFF) / 255f
            outPx[i] = labToRgb(lStar(v), aUp[i], bUp[i])
        }
        return Bitmap.createBitmap(outPx, w, h, Bitmap.Config.ARGB_8888)
    }

    // ----------------------------------------------------------------- color

    private fun toGrayscale(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val srcPx = IntArray(w * h)
        src.getPixels(srcPx, 0, w, 0, 0, w, h)
        val out = IntArray(w * h)
        for (i in srcPx.indices) {
            val p = srcPx[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // Rec.601 luma, matching cv2.COLOR_RGB2GRAY closely enough here.
            val y = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
            out[i] = (0xFF shl 24) or (y shl 16) or (y shl 8) or y
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    /** CIE L* (0..100) from a linear sRGB channel value in 0..1. */
    private fun lStar(v: Float): Float {
        val y = if (v <= 0.04045f) v / 12.92f
        else Math.pow(((v + 0.055f) / 1.055f).toDouble(), 2.4).toFloat()
        return if (y > 0.008856f) (116f * Math.cbrt(y.toDouble()).toFloat() - 16f)
        else 903.3f * y
    }

    private fun rgbToLab(r: Float, g: Float, b: Float): Triple<Float, Float, Float> {
        val rl = if (r <= 0.04045f) r / 12.92f else Math.pow(((r + 0.055f) / 1.055f).toDouble(), 2.4).toFloat()
        val gl = if (g <= 0.04045f) g / 12.92f else Math.pow(((g + 0.055f) / 1.055f).toDouble(), 2.4).toFloat()
        val bl = if (b <= 0.04045f) b / 12.92f else Math.pow(((b + 0.055f) / 1.055f).toDouble(), 2.4).toFloat()
        val x = (0.4124f * rl + 0.3576f * gl + 0.1805f * bl) / 0.95047f
        val y = (0.2126f * rl + 0.7152f * gl + 0.0722f * bl)
        val z = (0.0193f * rl + 0.1192f * gl + 0.9505f * bl) / 1.08883f
        fun f(t: Float): Float =
            if (t > 0.008856f) Math.cbrt(t.toDouble()).toFloat() else (7.787f * t + 16f / 116f)
        val fx = f(x); val fy = f(y); val fz = f(z)
        return Triple(116f * fy - 16f, 500f * (fx - fy), 200f * (fy - fz))
    }

    private fun labToRgb(L: Float, a: Float, b: Float): Int {
        val fy = (L + 16f) / 116f
        val fx = fy + a / 500f
        val fz = fy - b / 200f
        fun finv(t: Float): Float {
            val t3 = t * t * t
            return if (t3 > 0.008856f) t3 else (t - 16f / 116f) / 7.787f
        }
        val x = 0.95047f * finv(fx)
        val y = 1.0f * finv(fy)
        val z = 1.08883f * finv(fz)
        var r = 3.2406f * x - 1.5372f * y - 0.4986f * z
        var g = -0.9689f * x + 1.8758f * y + 0.0415f * z
        var bl = 0.0557f * x - 0.2040f * y + 1.0570f * z
        fun gamma(v: Float): Float {
            val c = v.coerceIn(0f, 1f)
            return if (c > 0.0031308f)
                (1.055f * Math.pow(c.toDouble(), 1.0 / 2.4).toFloat() - 0.055f)
            else 12.92f * c
        }
        r = gamma(r); g = gamma(g); bl = gamma(bl)
        val ri = (r * 255f).toInt().coerceIn(0, 255)
        val gi = (g * 255f).toInt().coerceIn(0, 255)
        val bi = (bl * 255f).toInt().coerceIn(0, 255)
        return (0xFF shl 24) or (ri shl 16) or (gi shl 8) or bi
    }

    private fun recentre(plane: FloatArray) {
        val copy = plane.copyOf()
        copy.sort()
        val median = copy[copy.size / 2]
        for (i in plane.indices) plane[i] -= median
    }

    /** Flatten a nested Java array (Float/float[]/Array<*>) into [dst]. */
    private fun flattenFloats(src: Any?, dst: FloatArray, offset: Int): Int {
        var idx = offset
        when (src) {
            is FloatArray -> for (v in src) { dst[idx++] = v }
            is Array<*> -> for (e in src) idx = flattenFloats(e, dst, idx)
            is Float -> { if (idx < dst.size) dst[idx++] = src }
            is Double -> { if (idx < dst.size) dst[idx++] = src.toFloat() }
            is Number -> { if (idx < dst.size) dst[idx++] = src.toFloat() }
        }
        return idx
    }

    /** Bilinear resize of a scalar plane with a gain and a ±60 clamp. */
    private fun resizePlane(src: FloatArray, sw: Int, sh: Int,
                            dw: Int, dh: Int, gain: Float): FloatArray {
        val out = FloatArray(dw * dh)
        for (y in 0 until dh) {
            val sy = (y.toFloat() * (sh - 1) / maxOf(1, dh - 1))
            val y0 = sy.toInt(); val y1 = minOf(y0 + 1, sh - 1)
            val wy = sy - y0
            for (x in 0 until dw) {
                val sx = (x.toFloat() * (sw - 1) / maxOf(1, dw - 1))
                val x0 = sx.toInt(); val x1 = minOf(x0 + 1, sw - 1)
                val wx = sx - x0
                val top = src[y0 * sw + x0] * (1 - wx) + src[y0 * sw + x1] * wx
                val bot = src[y1 * sw + x0] * (1 - wx) + src[y1 * sw + x1] * wx
                out[y * dw + x] = ((top * (1 - wy) + bot * wy) * gain).coerceIn(-60f, 60f)
            }
        }
        return out
    }
}

/** Single-channel guided filter (He et al.) with box blur. */
private object GuidedFilter {
    fun smooth(guide: FloatArray, src: FloatArray, w: Int, h: Int,
               radius: Int, eps: Float): FloatArray {
        val mi = boxBlur(guide, w, h, radius)
        val mp = boxBlur(src, w, h, radius)
        val gp = FloatArray(w * h) { guide[it] * src[it] }
        val mip = boxBlur(gp, w, h, radius)
        val gg = FloatArray(w * h) { guide[it] * guide[it] }
        val mi2 = boxBlur(gg, w, h, radius)
        val a = FloatArray(w * h)
        val b = FloatArray(w * h)
        for (i in a.indices) {
            val cov = mip[i] - mi[i] * mp[i]
            val v = mi2[i] - mi[i] * mi[i]
            a[i] = cov / (v + eps)
            b[i] = mp[i] - a[i] * mi[i]
        }
        val ma = boxBlur(a, w, h, radius)
        val mb = boxBlur(b, w, h, radius)
        return FloatArray(w * h) { ma[it] * guide[it] + mb[it] }
    }

    private fun boxBlur(src: FloatArray, w: Int, h: Int, r: Int): FloatArray {
        val tmp = FloatArray(w * h)
        val out = FloatArray(w * h)
        val win = 2 * r + 1
        for (y in 0 until h) {
            var sum = 0f
            for (i in -r..r) sum += src[y * w + i.coerceIn(0, w - 1)]
            for (x in 0 until w) {
                tmp[y * w + x] = sum / win
                val add = src[y * w + (x + r + 1).coerceIn(0, w - 1)]
                val sub = src[y * w + (x - r).coerceIn(0, w - 1)]
                sum += add - sub
            }
        }
        for (x in 0 until w) {
            var sum = 0f
            for (i in -r..r) sum += tmp[i.coerceIn(0, h - 1) * w + x]
            for (y in 0 until h) {
                out[y * w + x] = sum / win
                val add = tmp[(y + r + 1).coerceIn(0, h - 1) * w + x]
                val sub = tmp[(y - r).coerceIn(0, h - 1) * w + x]
                sum += add - sub
            }
        }
        return out
    }
}
