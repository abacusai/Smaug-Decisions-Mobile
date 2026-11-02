package ai.abacus.decide

import kotlin.math.exp

/** The decision readout on the letters' (already softcapped) logits, in float64 like the numpy reference. */
object Readout {
    /** one answer: softmax over the option letters' logits */
    fun softmax(z: DoubleArray): DoubleArray {
        val m = z.max(); val e = DoubleArray(z.size) { exp(z[it] - m) }; val s = e.sum()
        return DoubleArray(z.size) { e[it] / s }
    }

    /** select-all: independent sigmoid(z - b) per option; b = readout.json multi_bias_b (head.pt 'b') */
    fun sigmoid(z: DoubleArray, b: Double): DoubleArray = DoubleArray(z.size) { 1.0 / (1.0 + exp(-(z[it] - b))) }

    fun probabilities(z: DoubleArray, multi: Boolean, b: Double): DoubleArray = if (multi) sigmoid(z, b) else softmax(z)
}
