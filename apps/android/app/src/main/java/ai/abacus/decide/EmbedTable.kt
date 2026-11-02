package ai.abacus.decide

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * Smaug-Audio-Decisions-0.7B's token-embedding table (scaled, [vocab, 512]), memory-mapped. The app ships it as int8 with one fp32 scale per
 * row (../ios/tools/quantize_embed.py: 128 MB + 1 MB instead of 256 MB fp16), dequantised row by row on gather. The fp16 table
 * stays readable for the parity check and older asset sets.
 */
class EmbedTable private constructor(private val q: ByteBuffer?, private val scales: FloatBuffer?, private val h: ShortBuffer?) {
    val format: String = if (q != null) "int8" else "fp16"
    val vocab: Int = q?.let { it.capacity() / DIM } ?: (h!!.capacity() / DIM)

    init { if (q != null) require(scales!!.capacity() == vocab) { "int8 table has $vocab rows but ${scales.capacity()} scales" } }

    /** row [id] as float32 into dst[off until off + 512] */
    fun row(id: Int, dst: FloatArray, off: Int) {
        require(id in 0 until vocab) { "token id $id out of range" }
        val base = id * DIM
        if (q != null) {
            val s = scales!!.get(id)
            for (j in 0 until DIM) dst[off + j] = q.get(base + j) * s
        } else for (j in 0 until DIM) dst[off + j] = Half.toFloat(h!!.get(base + j))
    }

    companion object {
        const val DIM = 512
        const val INT8 = "embed_tokens_int8.bin"
        const val SCALES = "embed_tokens_int8_scales.bin"
        const val FP16 = "embed_tokens_f16.bin"

        /** int8 [vocab * 512] + float32 little-endian [vocab] */
        fun int8(table: ByteBuffer, scales: ByteBuffer) =
            EmbedTable(table, scales.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(), null)

        /** raw fp16 [vocab * 512] little-endian */
        fun fp16(table: ByteBuffer) = EmbedTable(null, null, table.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer())

        /** the int8 table in [dir] when present, else the fp16 one */
        fun fromDir(dir: File): EmbedTable =
            if (File(dir, INT8).exists()) int8(map(File(dir, INT8)), map(File(dir, SCALES))) else fp16(map(File(dir, FP16)))

        /** a read-only mapping that stays valid after the file is closed */
        fun map(f: File): ByteBuffer = RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
    }
}
