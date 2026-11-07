package eu.odysseus.diffusion

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * Real-ESRGAN x4plus Anime 6B via ONNX Runtime.
 *
 * A stripped-down 6-block RRDBNet (vs 23-block x4plus), ~17.9 MB, ~4x faster
 * on CPU. Optimized for anime/stylized images; negligible fidelity loss for
 * non-photographic inputs.
 *
 * Input: NCHW float32 RGB in [0, 1] range (name: "image.1")
 * Output: NCHW float32 RGB in [0, 1] range (name: "image")
 */
object RealEsrganOnnxEngine {
    private const val TAG = "RealEsrganOnnx"
    private const val MAX_SIDE = 8192
    private const val TILE = 256
    private const val OVERLAP = 16

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var modelDir: File? = null
    private var isReady = false
    @Volatile var lastError: String = ""

    fun init(ctx: Context, dir: File): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()
            modelDir = dir
            if (!dir.exists()) {
                Log.w(TAG, "model dir missing: ${dir.absolutePath}")
                return false
            }
            val modelFile = File(dir, "RealESRGAN_x4plus_anime_6B.onnx")
            if (!modelFile.isFile) {
                lastError = "model file missing: ${modelFile.absolutePath}"
                Log.w(TAG, lastError)
                return false
            }
            val opts = OrtSession.SessionOptions().apply {
                val threads = HardwareProfile.optimalThreads(ctx)
                try { setIntraOpNumThreads(threads) } catch (_: Exception) { /* cleanup */ }
                try { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT) } catch (_: Exception) { /* cleanup */ }
                val accel = AppConfig.acceleration.lowercase()
                if (accel != "cpu") {
                    if (accel == "gpu") {
                        try {
                            addNnapi()
                            Log.i(TAG, "NNAPI (GPU/NPU) provider enabled")
                        } catch (e: Exception) {
                            Log.w(TAG, "NNAPI unavailable: ${e.message}")
                            try {
                                addXnnpack(java.util.HashMap())
                                Log.i(TAG, "XNNPACK provider enabled")
                            } catch (e2: Exception) {
                                Log.w(TAG, "XNNPACK unavailable: ${e2.message}")
                            }
                        }
                    } else {
                        try {
                            addXnnpack(java.util.HashMap())
                            Log.i(TAG, "XNNPACK provider enabled")
                        } catch (e: Exception) {
                            Log.w(TAG, "XNNPACK unavailable: ${e.message}")
                            try {
                                addNnapi()
                                Log.i(TAG, "NNAPI provider enabled")
                            } catch (e2: Exception) {
                                Log.w(TAG, "NNAPI unavailable: ${e2.message}")
                            }
                        }
                    }
                }
            }
            session = env!!.createSession(modelFile.absolutePath, opts)
            isReady = true
            Log.i(TAG, "Real-ESRGAN Anime 6B ONNX ready (threads=${HardwareProfile.optimalThreads(ctx)})")
            true
        } catch (e: Exception) {
            lastError = "init: ${e.message}"
            Log.e(TAG, lastError, e)
            false
        }
    }

    fun release() {
        session?.close()
        session = null
        env?.close()
        env = null
        isReady = false
        lastError = ""
    }

    fun isReady(): Boolean = isReady

    /**
     * Upscale a bitmap 4x (the network is native 4x). For other scales, the
     * result is downsampled accordingly.
     */
    fun upscale(src: Bitmap, scale: Int): Bitmap? {
        if (!isReady || session == null || env == null) {
            lastError = "engine not ready"
            return null
        }
        if (scale !in 2..4) {
            lastError = "scale must be 2, 3, or 4"
            return null
        }
        val w = src.width
        val h = src.height
        if (w * scale > MAX_SIDE || h * scale > MAX_SIDE) {
            lastError = "upscaled ${w * scale}x${h * scale} exceeds cap ${MAX_SIDE}px"
            return null
        }

        // Run 4x, then downsample if a smaller scale was requested.
        val full = run4x(src)
        if (full == null) return null
        return if (scale == 4) full else {
            val scaled = Bitmap.createScaledBitmap(full, w * scale, h * scale, true)
            if (scaled !== full) full.recycle()
            scaled
        }
    }

    private fun run4x(src: Bitmap): Bitmap? {
        val w = src.width
        val h = src.height
        val outW = w * 4
        val outH = h * 4

        // Use tiling for large images to avoid OOM.
        if (w <= TILE && h <= TILE) {
            return runSingle(src, w, h)
        }

        val outPx = IntArray(outW * outH)
        var y = 0
        while (y < h) {
            val y1 = maxOf(y - OVERLAP, 0)
            val y2 = minOf(y + TILE + OVERLAP, h)
            var x = 0
            while (x < w) {
                val x1 = maxOf(x - OVERLAP, 0)
                val x2 = minOf(x + TILE + OVERLAP, w)
                val crop = Bitmap.createBitmap(src, x1, y1, x2 - x1, y2 - y1)
                val result = runSingle(crop, crop.width, crop.height)
                crop.recycle()
                if (result == null) {
                    lastError = "tile ${x},${y} failed"
                    return null
                }
                val resultPx = IntArray(result.width * result.height)
                result.getPixels(resultPx, 0, result.width, 0, 0, result.width, result.height)
                result.recycle()

                val cropOutW = (x2 - x1) * 4
                val cropOutH = (y2 - y1) * 4
                val ox = (x - x1) * 4
                val oy = (y - y1) * 4
                val cw = (minOf(x + TILE - OVERLAP, w) - x) * 4
                val ch = (minOf(y + TILE - OVERLAP, h) - y) * 4
                for (row in 0 until ch) {
                    System.arraycopy(
                        resultPx, row * cropOutW,
                        outPx, (oy + row) * outW + ox,
                        cw,
                    )
                }
                x += TILE - OVERLAP
            }
            y += TILE - OVERLAP
        }
        return Bitmap.createBitmap(outPx, outW, outH, Bitmap.Config.ARGB_8888)
    }

    private fun runSingle(src: Bitmap, srcW: Int, srcH: Int): Bitmap? {
        try {
            // Convert Bitmap to NCHW float32, normalized to [0, 1].
            val input = IntArray(srcW * srcH)
            src.getPixels(input, 0, srcW, 0, 0, srcW, srcH)
            val floatData = FloatArray(srcW * srcH * 3)
            var idx = 0
            for (i in input.indices) {
                val c = input[i]
                floatData[idx++] = ((c shr 16) and 0xFF).toFloat() / 255f // R
                floatData[idx++] = ((c shr 8) and 0xFF).toFloat() / 255f  // G
                floatData[idx++] = (c and 0xFF).toFloat() / 255f         // B
            }

            val inputShape = longArrayOf(1, 3, srcH, srcW)
            val inputTensor = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(floatData), inputShape)

            val feed = mapOf("image.1" to inputTensor)
            val results = session!!.run(feed)
            val outputTensor = results.values().iterator().next()
            val outputData = outputTensor.value?.asFloatBuffer()?.let { buf ->
                val arr = FloatArray(buf.remaining())
                buf.get(arr)
                arr
            } ?: run {
                lastError = "no output tensor"
                null
            }
            inputTensor.close()
            outputTensor.close()
            results.close()

            if (outputData == null) {
                lastError = "no output data"
                return null
            }

            // Output: NCHW [1, 3, H*4, W*4], convert to ARGB_8888 Bitmap.
            val outH = srcH * 4
            val outW = srcW * 4
            val outPx = IntArray(outW * outH)
            for (row in 0 until outH) {
                for (col in 0 until outW) {
                    val r = (outputData[(0 * outH + row) * outW + col] * 255f).coerceIn(0f, 255f).toInt()
                    val g = (outputData[(1 * outH + row) * outW + col] * 255f).coerceIn(0f, 255f).toInt()
                    val b = (outputData[(2 * outH + row) * outW + col] * 255f).coerceIn(0f, 255f).toInt()
                    outPx[row * outW + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            return Bitmap.createBitmap(outPx, outW, outH, Bitmap.Config.ARGB_8888)
        } catch (e: Exception) {
            lastError = "runSingle: ${e.message}"
            Log.e(TAG, lastError, e)
            return null
        }
    }
}
