package ai.abacus.decide

import java.io.File
import kotlin.math.tanh

/**
 * Smaug-Decisions on LiteRT-LM (CPU / XNNPack), the same .litertlm file and C call sequence as the iOS app (DecideShim.c): one engine,
 * a raw session per decision (no chat template; the runtime adds <bos>, so the prompt is passed without it), a multimodal prefill
 * from our harness patch (the <|audio|> / <|image|> placeholder is expanded by the model's own data processor from a WAV / PNG
 * file), one text-scoring step, then the raw logits of the option letters. The .litertlm graph returns raw LM-head logits, so
 * the softcap 30 is applied here; then softmax, or sigmoid(z - b) for select-all.
 *
 * [cacheDir] must be writable and kept between runs: LiteRT-LM writes its XNNPack weight cache there (about 2.8 GB) on the first
 * load; without it the engine holds about 4.8 GB of dirty memory instead of 0.8 GB (docs/devices.md). [tmpDir] receives the
 * per-decision WAV / PNG. [maxTokens] must match the file's KV cache (2048 for the Smaug-Decisions-LiteRT builds), and prompts are budgeted
 * to [promptTokens] (smaug_decide's 3072 would overflow a 2048 KV cache on long text). Runs on Android and the desktop JVM.
 *
 * Vision: the engine is created without the vision encoder ([vision] = ON_DEMAND, the default) and recreated with it when an image
 * decision arrives (about 1.5 s with a warm cache). [keepVision] = false recreates it without the encoder right after the image,
 * which gives back the encoder's activations (about 1.2 GB dirty at 560 soft tokens) on memory-tight devices.
 */
class LiteRtDecider(modelPath: String, cacheDir: File, private val tmpDir: File, threads: Int = 4, val maxTokens: Int = 2048,
                    val promptTokens: Int = PROMPT_TOKENS, val visualBudget: Int = Prompt.IMAGE_TOKENS,
                    val vision: Vision = Vision.ON_DEMAND, val keepVision: Boolean = true, val b: Double = Decider.DEFAULT_B,
                    /** false for a Smaug-Decisions variant without the audio encoder (smaug-decisions-noaudio-*, smaug-decisions-textonly) */
                    val audio: Boolean = true,
                    /** LiteRT-LM backend for the decoder: "cpu" (XNNPack, the Android default) or "gpu"; encoders stay on the CPU */
                    val engineBackend: String = "cpu") : BigModel {
    enum class Vision { ALWAYS, ON_DEMAND, NEVER }

    override val backend = "LiteRT-LM"
    private val modelPath = modelPath
    private val cachePath = cacheDir.path
    private val threads = threads
    private var handle: Long = 0L
    private var handleHasVision = false
    private val letterIds: IntArray
    var engineLoads = 0; private set
    var lastPositions = 0; private set
    var lastMs = 0f; private set

    init {
        cacheDir.mkdirs(); tmpDir.mkdirs()
        open(vision == Vision.ALWAYS)
        letterIds = IntArray(Prompt.LETTERS.length) { nativeSingleToken(handle, Prompt.LETTERS[it].toString()) }
        require(letterIds.all { it >= 0 }) { "a letter is not a single token: ${nativeLastError()}" }
    }

    private fun open(withVision: Boolean) {
        if (handle != 0L) nativeDelete(handle)
        handle = nativeCreate(modelPath, cachePath, maxTokens, threads, withVision, audio, engineBackend)
        require(handle != 0L) { "LiteRT-LM failed to load $modelPath: ${nativeLastError()}" }
        handleHasVision = withVision; engineLoads++
    }

    @Synchronized
    override fun evaluate(input: Decider.Input, question: String, options: List<String>, descriptions: List<String?>?,
                          multi: Boolean, rules: String?, context: String?): Decider.Result {
        check(handle != 0L) { "closed" }
        val isImage = input is Decider.Input.Image
        require(!isImage || vision != Vision.NEVER) { "this LiteRtDecider was created without vision" }
        if (isImage && !handleHasVision) open(true)
        require(options.size in 2..Prompt.LETTERS.length) { "2 to ${Prompt.LETTERS.length} options" }
        val desc = descriptions ?: List(options.size) { null }
        val audio = (input as? Decider.Input.Audio)?.let { AudioIn.lastSeconds(it.pcm16k) }
        val media = Prompt.mediaTokens(audio?.size, input is Decider.Input.Image)
        val tag = when (input) { is Decider.Input.Audio -> Prompt.AUDIO_TAG; is Decider.Input.Image -> Prompt.IMAGE_TAG; is Decider.Input.Text -> null }
        val prompt = Prompt.render(question, options, desc, multi, tag, rules, context, (input as? Decider.Input.Text)?.text,
            Prompt.charBudget(media, options.size, promptTokens))
        val mediaFile = when (input) {
            is Decider.Input.Audio -> File(tmpDir, "decide.wav").also { MediaFiles.writeWavFloat(it, audio!!) }
            is Decider.Input.Image -> File(tmpDir, "decide.png").also { MediaFiles.writePng(it, input.original) }
            is Decider.Input.Text -> null
        }
        val mediaJson = mediaFile?.let { """[{"type":"${if (input is Decider.Input.Audio) "audio" else "image"}","path":"${it.path.replace("\\", "\\\\").replace("\"", "\\\"")}"}]""" } ?: ""
        val r = nativeDecide(handle, prompt.removePrefix("<bos>").toByteArray(Charsets.UTF_8), mediaJson, visualBudget, letterIds.copyOf(options.size))
            ?: throw IllegalStateException("LiteRT-LM decision failed: ${nativeLastError()}")
        if (isImage && !keepVision && vision == Vision.ON_DEMAND) open(false)
        val n = options.size
        lastPositions = r[n + 1].toInt(); lastMs = r[n + 2] + r[n + 3]
        val z = DoubleArray(n) { tanh(r[it] / SOFTCAP) * SOFTCAP }
        return Decider.Result(prompt, z, Readout.probabilities(z, multi, b), lastPositions, lastMs)
    }

    @Synchronized override fun close() { if (handle != 0L) { nativeDelete(handle); handle = 0L } }

    companion object {
        const val SOFTCAP = 30.0
        /** prompt budget for the 2048-token KV builds (the rendered prompt + media must stay under the KV size) */
        const val PROMPT_TOKENS = 2000

        init { System.loadLibrary("decide_litert_jni") }
        @JvmStatic external fun nativeCreate(model: String, cacheDir: String?, maxTokens: Int, threads: Int, vision: Boolean, audio: Boolean, backend: String): Long
        @JvmStatic external fun nativeDelete(h: Long)
        @JvmStatic external fun nativeSingleToken(h: Long, s: String): Int
        @JvmStatic external fun nativeLastError(): String
        @JvmStatic external fun nativeDecide(h: Long, prompt: ByteArray, mediaJson: String, visualBudget: Int, ids: IntArray): FloatArray?
    }
}
