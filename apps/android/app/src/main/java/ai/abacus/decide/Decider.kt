package ai.abacus.decide

/**
 * Smaug-Decisions on llama.cpp: the training prompt ([Prompt], a port of smaug_decide.render), one prefill, the letters'
 * logits at the last position, softmax (one answer) or sigmoid(z - b) (select-all) ([Readout]). llama.cpp's gemma4 graph already
 * applies the final logit softcap (30), so it is not applied again. [b] is readout.json multi_bias_b (head.pt 'b', 8.0105 for
 * Smaug-Decisions). Runs on Android and on the desktop JVM (load libdecide_jni from java.library.path).
 */
class Decider(modelPath: String, mmprojPath: String?, val b: Double = DEFAULT_B, threads: Int = 4, gpuLayers: Int = 0, nCtx: Int = 4096,
              imageMaxTokens: Int = Prompt.IMAGE_TOKENS) : BigModel {
    override val backend = "llama.cpp"
    private var handle: Long = nativeLoad(modelPath, mmprojPath, threads, gpuLayers, nCtx, imageMaxTokens)
    private val letterIds: IntArray
    var lastPositions = 0; private set
    var lastMs = 0f; private set

    init {
        require(handle != 0L) { "llama.cpp failed to load $modelPath / $mmprojPath" }
        letterIds = IntArray(Prompt.LETTERS.length) { nativeTokenId(handle, Prompt.LETTERS[it].toString()) }
        require(letterIds.all { it >= 0 }) { "a letter is not a single token" }
        require(nativeSoftcap(handle) > 0f) { "GGUF has no final_logit_softcapping; apply tanh(z/30)*30 yourself" }
    }

    sealed class Input {
        data class Text(val text: String) : Input()
        /** mono 16 kHz float PCM (use [AudioIn.load] for a WAV file); the last 30 s are used */
        class Audio(val pcm16k: FloatArray) : Input()
        /** The decoded image as given (RGB). llama.cpp gets [grid] (resized to the HF grid: its own gemma4v preprocessing never
         *  upscales); LiteRT-LM gets the original as a PNG and resizes to its 560-token grid itself. */
        class Image(val original: HfImage.Rgb) : Input() {
            val grid: HfImage.Rgb by lazy { HfImage.toGrid(original) }
            val rgb: ByteArray get() = grid.rgb
            val width: Int get() = grid.width
            val height: Int get() = grid.height
        }

        companion object {
            /** any-size RGB; [Image.grid] is the Pillow-bicubic resize to transformers' Gemma4 grid (bit-exact with llama_decide.hf_resize) */
            fun fromRgb(rgb: ByteArray, width: Int, height: Int): Image = Image(HfImage.Rgb(rgb, width, height))

            /** Bitmap -> RGB bytes (alpha dropped) -> [fromRgb]. Decode with inPremultiplied = false for images with alpha. */
            fun fromBitmap(bmp: android.graphics.Bitmap): Image {
                val w = bmp.width; val h = bmp.height
                val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
                val rgb = ByteArray(w * h * 3)
                for (i in px.indices) { val c = px[i]; rgb[3 * i] = (c shr 16).toByte(); rgb[3 * i + 1] = (c shr 8).toByte(); rgb[3 * i + 2] = c.toByte() }
                return fromRgb(rgb, w, h)
            }
        }
    }

    /** Everything one decision produced: the rendered prompt, the letters' logits z, the probabilities, positions and time. */
    class Result(val prompt: String, val z: DoubleArray, val p: DoubleArray, val positions: Int, val ms: Float)

    override fun evaluate(input: Input, question: String, options: List<String>, descriptions: List<String?>?,
                          multi: Boolean, rules: String?, context: String?): Result {
        require(options.size in 2..Prompt.LETTERS.length) { "2 to ${Prompt.LETTERS.length} options" }
        val desc = descriptions ?: List(options.size) { null }
        val audio = (input as? Input.Audio)?.let { AudioIn.lastSeconds(it.pcm16k) }
        val media = Prompt.mediaTokens(audio?.size, input is Input.Image)
        val tag = when (input) { is Input.Audio -> Prompt.AUDIO_TAG; is Input.Image -> Prompt.IMAGE_TAG; is Input.Text -> null }
        val prompt = Prompt.render(question, options, desc, multi, tag, rules, context, (input as? Input.Text)?.text,
            Prompt.charBudget(media, options.size))
        val ids = letterIds.copyOf(options.size)
        val bytes = prompt.toByteArray(Charsets.UTF_8)
        val r = when (input) {
            is Input.Text -> nativeEval(handle, bytes, 0, null, null, 0, 0, ids)
            is Input.Image -> nativeEval(handle, bytes, 1, input.rgb, null, input.width, input.height, ids)
            is Input.Audio -> nativeEval(handle, bytes, 2, null, audio, 0, 0, ids)
        }
        lastPositions = r[options.size].toInt(); lastMs = r[options.size + 1]
        check(lastPositions > 0) { "dec_eval failed: $lastPositions" }
        val z = DoubleArray(options.size) { r[it].toDouble() }
        return Result(prompt, z, Readout.probabilities(z, multi, b), lastPositions, lastMs)
    }

    override fun close() { if (handle != 0L) { nativeFree(handle); handle = 0L } }

    companion object {
        /** readout.json multi_bias_b of the shipped Smaug-Decisions checkpoint */
        const val DEFAULT_B = 8.010554313659668

        init { System.loadLibrary("decide_jni") }
        @JvmStatic external fun nativeLoad(model: String, mmproj: String?, threads: Int, ngl: Int, nctx: Int, imageMaxTokens: Int): Long
        @JvmStatic external fun nativeFree(h: Long)
        @JvmStatic external fun nativeTokenId(h: Long, s: String): Int
        @JvmStatic external fun nativeSoftcap(h: Long): Float
        @JvmStatic external fun nativeEval(h: Long, prompt: ByteArray, media: Int, rgb: ByteArray?, pcm: FloatArray?, nx: Int, ny: Int, ids: IntArray): FloatArray
    }
}
