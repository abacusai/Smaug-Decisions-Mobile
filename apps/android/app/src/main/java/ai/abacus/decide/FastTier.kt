package ai.abacus.decide

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer

/**
 * The fast tier (Smaug-Audio-Decisions-0.7B, 744M) on ONNX Runtime: the marker readout of smaug_decide, as the iOS app does it with Core ML.
 * Pre-tokenised prompt ids ([Scenario]) + recorded audio -> [MelFrontEnd] -> audio.onnx -> splice the audio rows into the
 * token embeddings ([EmbedTable]: int8 + per-row scales, memory-mapped) -> trunk.onnx -> the linear head at each option's <unused0> marker -> softmax, or a
 * sigmoid per option for select-all. Both graphs take dynamic lengths (fast/export_onnx.py). Runs on Android (onnxruntime-android)
 * and on the desktop JVM (onnxruntime), same Java API.
 *
 * [source]: a directory with audio.onnx, trunk.onnx, the table (embed_tokens_int8.bin + embed_tokens_int8_scales.bin, or
 * embed_tokens_f16.bin) and head.json, or the same as buffers (on Android:
 * memory-mapped straight out of the APK, where they are stored uncompressed, so nothing is extracted to storage).
 */
class FastTier(source: Source, threads: Int = 4, xnnpack: Boolean = false) : AutoCloseable {
    constructor(dir: File, threads: Int = 4, xnnpack: Boolean = false) : this(Source.Dir(dir), threads, xnnpack)

    sealed class Source {
        class Dir(val dir: File) : Source()
        /** audio / trunk: ONNX bytes (a direct, e.g. memory-mapped, buffer); table: the token embeddings */
        class Buffers(val audio: java.nio.ByteBuffer, val trunk: java.nio.ByteBuffer, val table: EmbedTable, val headJson: String) : Source()
    }

    private val env = OrtEnvironment.getEnvironment()
    private val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (xnnpack) addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
    }
    private val audio: OrtSession
    private val trunk: OrtSession
    /** the token-embedding table (int8 in the shipped app) */
    val table: EmbedTable
    private val headW: FloatArray
    private val headB: Float

    init {
        val headText: String
        when (source) {
            is Source.Dir -> {
                audio = env.createSession(File(source.dir, "audio.onnx").path, opts)
                trunk = env.createSession(File(source.dir, "trunk.onnx").path, opts)
                table = EmbedTable.fromDir(source.dir)
                headText = File(source.dir, "head.json").readText()
            }
            is Source.Buffers -> {
                audio = env.createSession(source.audio, opts); trunk = env.createSession(source.trunk, opts)
                table = source.table; headText = source.headJson
            }
        }
        val head = JSONObject(headText)
        val w = head.getJSONArray("weight"); headW = FloatArray(w.length()) { w.getDouble(it).toFloat() }
        headB = head.getDouble("bias").toFloat()
        require(headW.size == HIDDEN) { "head weight ${headW.size} != $HIDDEN" }
    }

    class Result(val z: DoubleArray, val p: DoubleArray, val tokens: Int, val audioTokens: Int, val msMel: Double, val msAudio: Double,
                 val msTrunk: Double) { val msTotal get() = msMel + msAudio + msTrunk }

    /** An audio scenario on a 16 kHz mono clip (the last 30 s are used, like smaug_decide.load_audio). [table] overrides the
     *  embedding table (the int8 vs fp16 parity check). */
    fun run(s: Scenario, pcm16k: FloatArray, table: EmbedTable = this.table): Result {
        require(s.kind == "audio") { "the fast tier runs audio scenarios; ${s.id} is ${s.kind}" }
        val clip = AudioIn.lastSeconds(pcm16k)
        val t0 = System.nanoTime()
        val feats = MelFrontEnd.features(clip)
        val frames = feats.size / MelFrontEnd.NMELS
        require(frames > 0) { "clip too short (${clip.size} samples)" }
        val t1 = System.nanoTime()
        val emb = OnnxTensor.createTensor(env, FloatBuffer.wrap(feats), longArrayOf(1, frames.toLong(), MelFrontEnd.NMELS.toLong())).use { x ->
            audio.run(mapOf("features" to x)).use { r -> (r[0] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } } }
        }
        val t2 = System.nanoTime()
        val nAudio = MelFrontEnd.audioTokens(frames)
        check(emb.size >= nAudio * DIM) { "audio unit returned ${emb.size / DIM} rows < $nAudio" }
        val ids = s.preIds + IntArray(nAudio) { s.audioTokenId } + s.postIds
        val markers = s.markerOffsetsInPost.map { s.preIds.size + nAudio + it }
        val x = FloatArray(ids.size * DIM)
        var row = 0
        for ((i, id) in ids.withIndex()) {
            if (i >= s.preIds.size && row < nAudio) { System.arraycopy(emb, row * DIM, x, i * DIM, DIM); row++ } else table.row(id, x, i * DIM)
        }
        val (z, msTrunk) = trunkHead(x, ids.size, markers)
        return Result(z, probs(z, s.multi), ids.size, nAudio, (t1 - t0) / 1e6, (t2 - t1) / 1e6, msTrunk)
    }

    /** A text prompt already tokenised for the audio model (ids incl. <bos>, marker positions). Not used by the cascade (text goes to Smaug-Decisions). */
    fun runIds(ids: IntArray, markers: List<Int>, multi: Boolean): Result {
        val x = FloatArray(ids.size * DIM); ids.forEachIndexed { i, id -> table.row(id, x, i * DIM) }
        val (z, ms) = trunkHead(x, ids.size, markers)
        return Result(z, probs(z, multi), ids.size, 0, 0.0, 0.0, ms)
    }

    private fun trunkHead(x: FloatArray, n: Int, markers: List<Int>): Pair<DoubleArray, Double> {
        val t = System.nanoTime()
        val hidden = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, n.toLong(), DIM.toLong())).use { xe ->
            trunk.run(mapOf("inputs_embeds" to xe)).use { r -> (r[0] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } } }
        }
        val ms = (System.nanoTime() - t) / 1e6
        val z = DoubleArray(markers.size) { k ->
            var acc = headB.toDouble(); val off = markers[k] * HIDDEN
            for (j in 0 until HIDDEN) acc += hidden[off + j].toDouble() * headW[j]
            acc
        }
        return z to ms
    }

    override fun close() { audio.close(); trunk.close(); opts.close() }

    companion object {
        const val DIM = 512
        const val HIDDEN = 768

        /** marker readout (no bias b: the head's own bias is in z): softmax, or a sigmoid per option for select-all */
        fun probs(z: DoubleArray, multi: Boolean): DoubleArray = if (multi) Readout.sigmoid(z, 0.0) else Readout.softmax(z)
    }
}

/** IEEE 754 binary16 -> float32 (android.util.Half is not on the desktop JVM; Float.float16ToFloat needs Java 20). */
object Half {
    fun toFloat(h: Short): Float {
        val bits = h.toInt() and 0xffff
        val sign = (bits ushr 15) shl 31
        var exp = (bits ushr 10) and 0x1f
        var mant = bits and 0x3ff
        return when {
            exp == 0x1f -> java.lang.Float.intBitsToFloat(sign or 0x7f800000 or (mant shl 13))
            exp == 0 -> if (mant == 0) java.lang.Float.intBitsToFloat(sign) else {
                exp = 1
                while (mant and 0x400 == 0) { mant = mant shl 1; exp-- }
                mant = mant and 0x3ff
                java.lang.Float.intBitsToFloat(sign or ((exp + 112) shl 23) or (mant shl 13))
            }
            else -> java.lang.Float.intBitsToFloat(sign or ((exp + 112) shl 23) or (mant shl 13))
        }
    }
}
