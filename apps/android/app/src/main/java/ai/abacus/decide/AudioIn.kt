package ai.abacus.decide

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Audio input like llama_decide.load_audio: mono float32, resampled to 16 kHz with linear interpolation, last 30 s. Pure Kotlin. */
object AudioIn {
    class Wav(val samples: FloatArray, val sampleRate: Int)

    /** RIFF/WAVE reader: PCM 8/16/24/32-bit and IEEE float32, any channel count, mixed down to mono (soundfile scaling). */
    fun readWav(bytes: ByteArray): Wav {
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size >= 12 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" && String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") { "not a WAV file" }
        var pos = 12; var fmt = -1; var ch = 0; var sr = 0; var bits = 0; var dataOff = -1; var dataLen = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII); val len = bb.getInt(pos + 4)
            if (id == "fmt ") {
                fmt = bb.getShort(pos + 8).toInt() and 0xffff; ch = bb.getShort(pos + 10).toInt(); sr = bb.getInt(pos + 12); bits = bb.getShort(pos + 22).toInt()
                if (fmt == 0xFFFE && len >= 40) fmt = bb.getShort(pos + 32).toInt() and 0xffff   // WAVE_FORMAT_EXTENSIBLE sub-format
            } else if (id == "data") { dataOff = pos + 8; dataLen = minOf(len.toLong() and 0xffffffffL, (bytes.size - dataOff).toLong()).toInt(); break }
            pos += 8 + len + (len and 1)
        }
        require(fmt >= 0 && dataOff >= 0 && ch > 0) { "WAV without fmt/data chunk" }
        val bps = bits / 8; val n = dataLen / (bps * ch)
        val out = FloatArray(n)
        for (i in 0 until n) {
            var acc = 0f
            for (c in 0 until ch) {
                val p = dataOff + (i * ch + c) * bps
                val v: Float = when {
                    fmt == 3 && bits == 32 -> bb.getFloat(p)
                    fmt == 1 && bits == 16 -> bb.getShort(p) / 32768f
                    fmt == 1 && bits == 8 -> ((bytes[p].toInt() and 0xff) - 128) / 128f
                    fmt == 1 && bits == 24 -> ((bytes[p].toInt() and 0xff) or ((bytes[p + 1].toInt() and 0xff) shl 8) or (bytes[p + 2].toInt() shl 16)) / 8388608f
                    fmt == 1 && bits == 32 -> (bb.getInt(p) / 2147483648.0).toFloat()
                    else -> throw IllegalArgumentException("unsupported WAV format $fmt / $bits bits")
                }
                acc = if (c == 0) v else acc + v
            }
            out[i] = if (ch == 1) acc else acc / ch
        }
        return Wav(out, sr)
    }

    /** np.interp(np.linspace(0, len - 1, n), np.arange(len), w) with n = round(len * 16000 / sr) (half to even), as float32. */
    fun resampleTo16k(w: FloatArray, sr: Int): FloatArray {
        if (sr == Prompt.SAMPLE_RATE) return w
        val n = Math.rint(w.size.toDouble() * Prompt.SAMPLE_RATE / sr).toInt()
        val stop = (w.size - 1).toDouble(); val div = n - 1
        val step = if (div > 0) stop / div else Double.NaN
        val out = FloatArray(n)
        val last = w.size - 1
        for (i in 0 until n) {
            var x = if (div > 0) (if (step == 0.0) i.toDouble() / div * stop else i * step) else 0.0
            if (div > 0 && i == n - 1) x = stop          // linspace endpoint
            val r: Double = when {
                x >= last -> w[last].toDouble()
                x <= 0.0 -> w[0].toDouble()
                else -> {
                    val j = x.toInt()                        // xp = 0..len-1, so the bin is floor(x)
                    if (j.toDouble() == x) w[j].toDouble()
                    else { val slope = w[j + 1].toDouble() - w[j].toDouble(); slope * (x - j) + w[j] }
                }
            }
            out[i] = r.toFloat()
        }
        return out
    }

    /** the last 30 s, like w[-int(30 * 16000):] */
    fun lastSeconds(w: FloatArray, maxSamples: Int = Prompt.MAX_AUDIO_SAMPLES): FloatArray =
        if (w.size > maxSamples) w.copyOfRange(w.size - maxSamples, w.size) else w

    /** load_audio for a WAV file's bytes */
    fun load(bytes: ByteArray): FloatArray { val w = readWav(bytes); return lastSeconds(resampleTo16k(w.samples, w.sampleRate)) }
}
