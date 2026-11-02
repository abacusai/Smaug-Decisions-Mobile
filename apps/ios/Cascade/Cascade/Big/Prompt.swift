import Foundation

/// Port of `render` / `trim` and the char budget of `SmaugDecide.__call__` in smaug_decide.py (abacusai/Smaug-Decisions), verbatim.
/// Pure Swift, so the unit tests check it byte for byte against goldens rendered by the Python reference.
enum Prompt {
    static let letters = Array("ABCDEFGHIJKLMNOPQRST")
    static let sampleRate = 16_000
    static let maxAudioSeconds = 30
    static let maxTokens = 3072
    /// The on-device budget: the Smaug-Decisions-LiteRT builds hold 2048 tokens, so prompts are rendered as if MAX_TOKENS were 2000 (smaug_decide's
    /// formula and trim, smaller budget), leaving room for the scored answer token.
    static let deviceMaxTokens = 2000
    static let audioTokensPerSecond = 25
    static let imageTokens = 560
    static let audioTag = "<|audio|>"
    static let imageTag = "<|image|>"
    static let bos = "<bos>"
    static let preamble = "You make decisions for apps. Read the rules, the context and the input, then answer the question by choosing "
        + "from the lettered options."

    enum Input {
        case text(String)
        case image
        case audio(samples: Int)
    }

    /// Python's trim(): slices by code points with int() truncation; a zero tail keeps the whole string, like s[-0:].
    static func trim(_ s: String, _ n: Int) -> String {
        let cps = Array(s.unicodeScalars)
        if cps.count <= n { return s }
        let head = min(Int(Double(n) * 0.7), cps.count)
        let tail = Int(Double(n) * 0.3)
        let end = tail == 0 ? cps[...] : cps[max(cps.count - tail, 0)...]
        return string(cps[..<head]) + " ... " + string(end)
    }

    /// The training prompt: preamble, optional rules / context, the input, the lettered options, the answer instruction.
    static func render(question: String, options: [String], descriptions: [String?], multi: Bool, mediaTag: String?,
                       rules: String?, context: String?, textInput: String?, charBudget: Int) -> String {
        var parts = [preamble]
        if let rules, !rules.isEmpty { parts.append("Rules: \(trim(rules, floorDiv(charBudget, 2)))") }
        if let context, !context.isEmpty { parts.append("Context: \(trim(context, charBudget))") }
        if let mediaTag, !mediaTag.isEmpty {
            parts.append("Input: \(mediaTag)")
        } else if let textInput, !textInput.isEmpty {
            parts.append("Input: \(trim(textInput, charBudget))")
        }
        var lines = ["Question: \(question)" + (multi ? " Select all that apply." : "")]
        for (k, (option, desc)) in zip(options, descriptions).enumerated() {   // zip stops at the shorter list, like Python
            var line = "\(letters[k])) \(option)"
            if let desc, !desc.isEmpty { line += " \u{2014} \(desc)" }
            lines.append(line)
        }
        parts.append(lines.joined(separator: "\n"))
        parts.append(multi ? "Answer with the letters of all options that apply." : "Answer with the letter of the best option.")
        return "<bos><|turn>user\n" + parts.joined(separator: "\n") + "<turn|>\n<|turn>model\n"
    }

    /// Media tokens SmaugDecide budgets for: audio = seconds * 25 + 2, image = 560 + 2, text = 0.
    static func mediaTokens(audioSamples: Int?, image: Bool, imageTokens: Int = imageTokens) -> Double {
        if let audioSamples { return Double(audioSamples) / Double(sampleRate) * Double(audioTokensPerSecond) + 2 }
        return image ? Double(imageTokens + 2) : 0
    }

    /// int(max(400, (MAX_TOKENS - media_tokens - 80 - 10 * n_options) * 3.5))
    static func charBudget(mediaTokens: Double, options: Int, maxTokens: Int = maxTokens) -> Int {
        Int(max(400, (Double(maxTokens) - mediaTokens - 80 - Double(10 * options)) * 3.5))
    }

    /// The char budget assumes 3.5 characters per token; dense text (digits, other scripts) can need more tokens. Renders with a
    /// smaller MAX_TOKENS until the counted prompt fits `limit`. `render(m)` renders at MAX_TOKENS m, `count` returns its tokens.
    static func fitted(limit: Int, render: (Int) -> String, count: (String) throws -> Int) rethrows -> (prompt: String, tokens: Int, maxTokens: Int) {
        var m = limit
        var prompt = render(m), n = try count(prompt)
        for _ in 0..<8 where n > limit {
            m -= (n - limit) + 16
            prompt = render(m); n = try count(prompt)
        }
        return (prompt, n, m)
    }

    /// The full prompt for one question, as SmaugDecide.__call__ builds it (descriptions optional, one per option).
    static func decision(question: String, options: [String], multi: Bool, input: Input, rules: String? = nil,
                         context: String? = nil, descriptions: [String?]? = nil, maxTokens: Int = maxTokens,
                         imageTokens: Int = imageTokens) -> String {
        var tag: String? = nil
        var textInput: String? = nil
        var media = 0.0
        switch input {
        case .text(let t):
            textInput = t
        case .image:
            tag = imageTag
            media = mediaTokens(audioSamples: nil, image: true, imageTokens: imageTokens)
        case .audio(let n):
            tag = audioTag
            media = mediaTokens(audioSamples: n, image: false)
        }
        return render(question: question, options: options, descriptions: descriptions ?? Array(repeating: nil, count: options.count),
                      multi: multi, mediaTag: tag, rules: rules, context: context, textInput: textInput,
                      charBudget: charBudget(mediaTokens: media, options: options.count, maxTokens: maxTokens))
    }

    /// LiteRT-LM prepends BOS itself, so the rendered prompt goes in without its leading "<bos>".
    static func stripBOS(_ prompt: String) -> String { prompt.hasPrefix(bos) ? String(prompt.dropFirst(bos.count)) : prompt }

    /// Python's // (floors toward negative infinity).
    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
    }

    private static func string<S: Sequence>(_ scalars: S) -> String where S.Element == Unicode.Scalar {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }
}
