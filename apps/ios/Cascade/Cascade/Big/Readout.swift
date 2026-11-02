import Foundation

/// The decision readout on the option letters' logits, in Double like the numpy reference.
enum Readout {
    /// Gemma's final logit softcap. The .litertlm graph outputs raw LM-head logits, so it is applied here.
    static let softcap = 30.0
    /// The learned select-all threshold b (readout.json multi_bias_b of the Smaug-Decisions checkpoint).
    static let multiBias = 8.010554313659668

    static func capped(_ z: [Double], cap: Double = softcap) -> [Double] { z.map { tanh($0 / cap) * cap } }

    /// One answer: softmax over the option letters.
    static func softmax(_ z: [Double]) -> [Double] {
        guard let m = z.max() else { return [] }
        let e = z.map { exp($0 - m) }
        let s = e.reduce(0, +)
        return e.map { $0 / s }
    }

    /// Select-all: an independent sigmoid(z - b) per option.
    static func sigmoid(_ z: [Double], b: Double = multiBias) -> [Double] { z.map { 1 / (1 + exp(-($0 - b))) } }

    /// `z` must already be softcapped.
    static func probabilities(_ z: [Double], multi: Bool, b: Double = multiBias) -> [Double] {
        multi ? sigmoid(z, b: b) : softmax(z)
    }
}
