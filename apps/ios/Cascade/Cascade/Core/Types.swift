import Foundation

/// A pre-tokenised fast-tier prompt (tools/make_cases.py; the phone has no tokenizer for the audio model): ids before and after the audio block
/// and the option markers' offsets in `post_ids`.
struct FastPrompt: Codable, Hashable, Sendable {
    let pre_ids: [Int], post_ids: [Int], marker_offsets_in_post: [Int]
    let text: String
    var ref_torch: [Double]? = nil, ref_coreml_gpu: [Double]? = nil, ref_coreml_cpu: [Double]? = nil
    var ref_tokens: Int? = nil, window_s: Double? = nil
}

struct FastResult: Sendable {
    let probabilities: [Double]
    let tokens: Int, audioTokens: Int, audioSeconds: Double
    let msAudio: Double, msTrunk: Double, msTotal: Double
    var logits: [Double] = []
    var diagnostics = ""
}

struct MediaFile: Sendable {
    enum Kind: String, Sendable {
        case image, audio
    }

    let kind: Kind
    /// A 16 kHz mono float32 WAV, or an RGB PNG/JPEG, on disk (the runtime reads media by path).
    let path: String
}

struct BigResult: Sendable {
    let probabilities: [Double]
    /// Softcapped letter logits (patched runtime only).
    let logits: [Double]?
    let promptTokens: Int
    /// True when the count is the runtime's own prefill count (media tokens included), false for a text-token estimate.
    let tokensFromRuntime: Bool
    let prefillMs: Double
    let readoutMs: Double
    let patched: Bool
}

struct DeciderError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
