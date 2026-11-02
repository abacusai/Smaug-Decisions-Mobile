package ai.abacus.decide

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32
import java.util.zip.Deflater

/** Media files for LiteRT-LM's data processor, in pure Kotlin (Android and desktop JVM): 32-bit float WAV (what score.py / the
 *  iOS app write with soundfile / AVAudioFile) and an 8-bit RGB PNG (lossless, so the runtime decodes exactly our pixels). */
object MediaFiles {
    fun writeWavFloat(f: File, pcm: FloatArray, sampleRate: Int = 16000) {
        val data = pcm.size * 4
        val bb = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()).putInt(36 + data).put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()).putInt(16).putShort(3).putShort(1).putInt(sampleRate).putInt(sampleRate * 4).putShort(4).putShort(32)
        bb.put("data".toByteArray()).putInt(data)
        pcm.forEach { bb.putFloat(it) }
        f.writeBytes(bb.array())
    }

    fun writePng(f: File, img: HfImage.Rgb) {
        val raw = ByteArray((img.width * 3 + 1) * img.height)
        for (y in 0 until img.height) {
            raw[y * (img.width * 3 + 1)] = 0   // filter: none
            System.arraycopy(img.rgb, y * img.width * 3, raw, y * (img.width * 3 + 1) + 1, img.width * 3)
        }
        val z = Deflater(6).run { setInput(raw); finish(); val o = ByteArrayOutputStream(); val buf = ByteArray(1 shl 16)
            while (!finished()) o.write(buf, 0, deflate(buf)); end(); o.toByteArray() }
        val out = ByteArrayOutputStream(); val d = DataOutputStream(out)
        out.write(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 13, 10, 26, 10))
        fun chunk(type: String, body: ByteArray) {
            d.writeInt(body.size); val t = type.toByteArray(); d.write(t); d.write(body)
            d.writeInt(CRC32().apply { update(t); update(body) }.value.toInt())
        }
        chunk("IHDR", ByteBuffer.allocate(13).putInt(img.width).putInt(img.height).put(8).put(2).put(0).put(0).put(0).array())
        chunk("IDAT", z); chunk("IEND", ByteArray(0))
        f.writeBytes(out.toByteArray())
    }
}
