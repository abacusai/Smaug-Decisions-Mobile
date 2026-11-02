package ai.abacus.decide

/**
 * The Smaug-Decisions training prompt: a port of smaug_decide.render / trim and of the char budget in SmaugDecide.__call__. Pure Kotlin
 * (no Android or native dependencies) so JVM unit tests can check it byte for byte against the Python reference.
 */
object Prompt {
    const val LETTERS = "ABCDEFGHIJKLMNOPQRST"
    const val SAMPLE_RATE = 16000
    const val MAX_AUDIO_SAMPLES = 30 * SAMPLE_RATE
    /** Smaug-Decisions' reference smaug_decide.py on Hugging Face: 3072-token budget, images at 560 soft tokens */
    const val MAX_TOKENS = 3072
    const val AUDIO_TOKENS_PER_SECOND = 25
    const val IMAGE_TOKENS = 560
    const val AUDIO_TAG = "<|audio|>"
    const val IMAGE_TAG = "<|image|>"
    const val PREAMBLE = "You make decisions for apps. Read the rules, the context and the input, then answer the question by choosing " +
        "from the lettered options."

    /** Python's trim(): slices by code points, int() truncation. A zero tail keeps the whole string, like Python's s[-0:]. */
    fun trim(s: String, n: Int): String {
        val cps = s.codePoints().toArray()
        if (cps.size <= n) return s
        val head = minOf((n * 0.7).toInt(), cps.size)
        val tail = (n * 0.3).toInt()
        fun str(a: IntArray) = String(a, 0, a.size)
        val end = if (tail == 0) cps else cps.copyOfRange(maxOf(cps.size - tail, 0), cps.size)
        return str(cps.copyOfRange(0, head)) + " ... " + str(end)
    }

    /** Port of smaug_decide.render: the training prompt, verbatim. */
    fun render(question: String, options: List<String>, descriptions: List<String?>, multi: Boolean, mediaTag: String?,
               rules: String?, context: String?, textInput: String?, charBudget: Int): String {
        val parts = mutableListOf(PREAMBLE)
        if (!rules.isNullOrEmpty()) parts += "Rules: ${trim(rules, Math.floorDiv(charBudget, 2))}"
        if (!context.isNullOrEmpty()) parts += "Context: ${trim(context, charBudget)}"
        if (!mediaTag.isNullOrEmpty()) parts += "Input: $mediaTag"
        else if (!textInput.isNullOrEmpty()) parts += "Input: ${trim(textInput, charBudget)}"
        val lines = mutableListOf("Question: $question" + if (multi) " Select all that apply." else "")
        // zip() like Python: stops at the shorter list
        for (k in 0 until minOf(options.size, descriptions.size)) {
            lines += "${LETTERS[k]}) ${options[k]}" + (descriptions[k]?.takeIf { it.isNotEmpty() }?.let { " — $it" } ?: "")
        }
        parts += lines.joinToString("\n")
        parts += if (multi) "Answer with the letters of all options that apply." else "Answer with the letter of the best option."
        return "<bos><|turn>user\n" + parts.joinToString("\n") + "<turn|>\n<|turn>model\n"
    }

    /** media tokens SmaugDecide budgets for: audio = seconds * 25 + 2, image = 560 + 2, text = 0. */
    fun mediaTokens(audioSamples: Int?, image: Boolean): Double = when {
        audioSamples != null -> audioSamples.toDouble() / SAMPLE_RATE * AUDIO_TOKENS_PER_SECOND + 2
        image -> (IMAGE_TOKENS + 2).toDouble()
        else -> 0.0
    }

    /** int(max(400, (MAX_TOKENS - media_tokens - 80 - 10 * n_options) * 3.5)) */
    fun charBudget(mediaTokens: Double, nOptions: Int, maxTokens: Int = MAX_TOKENS): Int =
        maxOf(400.0, (maxTokens - mediaTokens - 80 - 10 * nOptions) * 3.5).toInt()
}
