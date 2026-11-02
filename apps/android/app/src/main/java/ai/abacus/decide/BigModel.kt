package ai.abacus.decide

/** The big tier (Smaug-Decisions) behind one interface: [LiteRtDecider] (LiteRT-LM, the default) or [Decider] (llama.cpp, fallback). */
interface BigModel : AutoCloseable {
    val backend: String

    /** One decision: the rendered training prompt, one prefill, the option letters' logits, probabilities ([Decider.Result]). */
    fun evaluate(input: Decider.Input, question: String, options: List<String> = listOf("Yes", "No"), descriptions: List<String?>? = null,
                 multi: Boolean = false, rules: String? = null, context: String? = null): Decider.Result

    fun decide(input: Decider.Input, question: String, options: List<String> = listOf("Yes", "No"), descriptions: List<String?>? = null,
               multi: Boolean = false, rules: String? = null, context: String? = null): Map<String, Float> =
        options.zip(evaluate(input, question, options, descriptions, multi, rules, context).p.map { it.toFloat() }).toMap()
}
