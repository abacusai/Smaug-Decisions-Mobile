package ai.abacus.decide

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Log-mel front end of the EmbeddingGemma 2 / Gemma 4 audio tower (transformers' Gemma4AudioFeatureExtractor defaults), a port of
 * the iOS app's MelFrontEnd.swift and of fast/mel.py: 16 kHz mono, 160 zeros prepended, frames of 320 samples every
 * 160, periodic Hann, |rFFT 512| (magnitude, not power), 128 HTK mel filters 0-8 kHz without normalisation, log(mel + 1e-3).
 * Only the valid frames are returned (frame i is valid when its 321-sample unfold window lies inside the audio), because the
 * fast tier's audio.onnx takes the unpadded clip. Computed in float64, returned as float32. Pure Kotlin (JVM-testable).
 */
object MelFrontEnd {
    const val FRAME = 320
    const val HOP = 160
    const val NFFT = 512
    const val NMELS = 128
    const val FLOOR = 1e-3
    private const val NFREQ = NFFT / 2 + 1

    fun frameCount(samples: Int): Int = max(0, Math.floorDiv(samples + FRAME / 2 - (FRAME + 1), HOP) + 1)

    /** audio tokens the tower makes from F frames: two stride-2 convolutions, ceil twice (capped at 750 = 30 s) */
    fun audioTokens(frames: Int): Int { var t = frames; repeat(2) { t = Math.floorDiv(t + 2 - 3, 2) + 1 }; return min(max(t, 0), 750) }

    private val window = DoubleArray(FRAME) { 0.5 - 0.5 * cos(2 * PI * it / FRAME) }

    /** [NFREQ * NMELS] row = frequency bin (transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk")) */
    private val filters: DoubleArray = run {
        fun hz2mel(f: Double) = 2595.0 * log10(1.0 + f / 700.0)
        fun mel2hz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)
        val lo = hz2mel(0.0); val hi = hz2mel(8000.0)
        val pts = DoubleArray(NMELS + 2) { mel2hz(lo + (hi - lo) * it / (NMELS + 1)) }
        val fb = DoubleArray(NFREQ * NMELS)
        for (m in 0 until NMELS) for (k in 0 until NFREQ) {
            val f = k * 16000.0 / (2 * (NFREQ - 1))
            val up = (f - pts[m]) / (pts[m + 1] - pts[m]); val down = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1])
            fb[k * NMELS + m] = max(0.0, min(up, down))
        }
        fb
    }

    // radix-2 complex FFT tables for n = 512
    private val bitrev = IntArray(NFFT) { i -> Integer.reverse(i) ushr (32 - 9) }
    private val cosT = DoubleArray(NFFT / 2) { cos(2 * PI * it / NFFT) }
    private val sinT = DoubleArray(NFFT / 2) { -kotlin.math.sin(2 * PI * it / NFFT) }

    private fun fft(re: DoubleArray, im: DoubleArray) {
        for (i in 0 until NFFT) { val j = bitrev[i]; if (j > i) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t } }
        var len = 2
        while (len <= NFFT) {
            val half = len / 2; val step = NFFT / len
            var s = 0
            while (s < NFFT) {
                for (k in 0 until half) {
                    val wr = cosT[k * step]; val wi = sinT[k * step]
                    val a = s + k; val b = a + half
                    val xr = re[b] * wr - im[b] * wi; val xi = re[b] * wi + im[b] * wr
                    re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi
                }
                s += len
            }
            len *= 2
        }
    }

    /** float32 16 kHz mono -> row-major [frames * 128] float32, frames = [frameCount] */
    fun features(samples: FloatArray): FloatArray {
        val n = frameCount(samples.size)
        val out = FloatArray(n * NMELS)
        val re = DoubleArray(NFFT); val im = DoubleArray(NFFT); val mag = DoubleArray(NFREQ)
        for (i in 0 until n) {
            java.util.Arrays.fill(re, 0.0); java.util.Arrays.fill(im, 0.0)
            for (j in 0 until FRAME) {
                val src = i * HOP + j - FRAME / 2          // index into the clip after prepending FRAME/2 zeros
                if (src >= 0 && src < samples.size) re[j] = samples[src].toDouble() * window[j]
            }
            fft(re, im)
            for (k in 0 until NFREQ) mag[k] = sqrt(re[k] * re[k] + im[k] * im[k])
            for (m in 0 until NMELS) {
                var acc = 0.0
                for (k in 0 until NFREQ) acc += mag[k] * filters[k * NMELS + m]
                out[i * NMELS + m] = ln(acc + FLOOR).toFloat()
            }
        }
        return out
    }
}
