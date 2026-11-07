package eu.odysseus.diffusion

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * On-device video re-encode (compress) using MediaCodec + ByteBuffer only.
 *
 * No EGL/GL bridge: frames are passed as raw ByteBuffers between the decoder
 * and encoder. This avoids GPU overhead, avoids thermal throttling, and works
 * on devices without a working Vulkan/GL driver.
 *
 * Pipeline:
 *   MediaExtractor → MediaCodec decoder (ByteBuffer in/out) → MediaCodec encoder (ByteBuffer in/out) → MediaMuxer
 *
 * Best-effort: any failure is converted into [MediaTools.Unsupported] so the
 * API answers an honest 503 instead of crashing the server process.
 */
object VideoTranscoder {

    private const val MAX_SIDE = 8192

    fun compress(input: File, preset: String): MediaTools.Output {
        val targetSizeMb = when (preset) {
            "web-720p", "mp4-small" -> 5f
            else -> 10f
        }

        val extractor = MediaExtractor()
        var muxer: MediaMuxer? = null
        var decoder: MediaCodec? = null
        var encoder: MediaCodec? = null
        val outFile = File(input.parentFile, "mt_compress.mp4")

        try {
            extractor.setDataSource(input.absolutePath)

            // Find video track
            var videoTrackIndex = -1
            var videoFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("video/")) {
                    videoTrackIndex = i
                    videoFormat = f
                    break
                }
            }
            if (videoTrackIndex < 0 || videoFormat == null)
                throw MediaTools.Unsupported("no video track")

            val mime = videoFormat.getString(MediaFormat.KEY_MIME)!!
            val width = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
            val height = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
            val frameRate = try { videoFormat.getInteger(MediaFormat.KEY_FRAME_RATE) } catch (_: Exception) { 30 }

            var durationUs = try { videoFormat.getLong(MediaFormat.KEY_DURATION) } catch (_: Exception) { 0L }
            if (durationUs <= 0) {
                extractor.selectTrack(videoTrackIndex)
                extractor.seekTo(Long.MAX_VALUE, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                val lastPts = extractor.sampleTime
                if (lastPts > 0) durationUs = lastPts
            }
            if (durationUs <= 0) throw MediaTools.Unsupported("could not determine video duration")

            val durationS = durationUs / 1_000_000.0
            val targetBits = targetSizeMb * 1024 * 1024 * 8
            val audioBitrate = 128000
            val videoBitrate = ((targetBits / durationS) - audioBitrate).toInt().coerceAtLeast(100000)

            extractor.selectTrack(videoTrackIndex)

            // --- Decoder: ByteBuffer input/output (no Surface) ---
            decoder = MediaCodec.createDecoderByType(mime)
            decoder!!.configure(videoFormat, null, null, 0)
            decoder!!.start()

            // --- Encoder: ByteBuffer input/output (no Surface) ---
            val encoderFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
            encoderFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            encoderFormat.setInteger(MediaFormat.KEY_BIT_RATE, videoBitrate)
            encoderFormat.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            encoderFormat.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            encoder!!.configure(encoderFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder!!.start()

            // --- Muxer ---
            muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

            var videoMuxTrack = -1
            var muxStarted = false
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                // === DECODER INPUT: feed compressed frames ---
                if (!inputDone) {
                    val inIdx = decoder!!.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = decoder!!.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            // End of stream
                            decoder!!.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val pts = extractor.sampleTime
                            decoder!!.queueInputBuffer(inIdx, 0, size, pts, 0)
                            extractor.advance()
                        }
                    }
                }

                // --- DECODER OUTPUT: get decoded YUV frames ---
                val decInfo = MediaCodec.BufferInfo()
                val decIdx = decoder!!.dequeueOutputBuffer(decInfo, 10_000)
                if (decIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    // Decoder output format changed - we ignore since we use fixed format
                } else if (decIdx >= 0) {
                    val decBuf = decoder!!.getOutputBuffer(decIdx)!!
                    val decSize = decBuf.limit() - decBuf.position()
                    decBuf.position(decBuf.position())  // position at start of data
                    decBuf.limit(decBuf.capacity())  // ensure full buffer readable

                    if (decSize > 0) {
                        // Feed decoded frame to encoder input buffer
                        val encInIdx = encoder!!.dequeueInputBuffer(0)
                        if (encInIdx >= 0) {
                            val encInBuf = encoder!!.getInputBuffer(encInIdx)!!
                            encInBuf.clear()
                            // Copy decoded YUV data to encoder input
                            val data = ByteArray(decSize)
                            decBuf.get(data)
                            encInBuf.put(data)
                            encoder!!.queueInputBuffer(encInIdx, 0, decSize, decInfo.presentationTimeUs, decInfo.flags)
                        }

                        decoder!!.releaseOutputBuffer(decIdx, false)

                        if (decInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            encoder!!.signalEndOfInputStream()
                        }
                    }
                }

                // --- ENCODER OUTPUT: get encoded H.264 frames ---
                val encIdx = encoder!!.dequeueOutputBuffer(bufferInfo, 10_000)
                if (encIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val newFormat = encoder!!.outputFormat
                    videoMuxTrack = muxer!!.addTrack(newFormat)
                    muxer!!.start()
                    muxStarted = true
                } else if (encIdx >= 0) {
                    val buf = encoder!!.getOutputBuffer(encIdx)!!
                    if (muxStarted && bufferInfo.size > 0) {
                        buf.position(bufferInfo.offset)
                        buf.limit(bufferInfo.offset + bufferInfo.size)
                        muxer!!.writeSampleData(videoMuxTrack, buf, bufferInfo)
                    }
                    encoder!!.releaseOutputBuffer(encIdx, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                }
            }

            val bytes = outFile.readBytes()
            outFile.delete()
            return MediaTools.Output(bytes, "compressed.mp4", "video/mp4")

        } catch (e: Exception) {
            try { outFile.delete() } catch (_: Exception) { /* cleanup */ }
            if (e is MediaTools.Unsupported) throw e
            throw MediaTools.Unsupported("video re-encode failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            try { decoder?.stop() } catch (_: Exception) { /* cleanup */ }
            try { decoder?.release() } catch (_: Exception) { /* cleanup */ }
            try { encoder?.stop() } catch (_: Exception) { /* cleanup */ }
            try { encoder?.release() } catch (_: Exception) { /* cleanup */ }
            try { muxer?.stop() } catch (_: Exception) { /* cleanup */ }
            try { muxer?.release() } catch (_: Exception) { /* cleanup */ }
            try { extractor.release() } catch (_: Exception) { /* cleanup */ }
        }
    }
}
