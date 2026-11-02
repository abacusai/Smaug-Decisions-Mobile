import Foundation

/// cascade.json (tools/make_cases.py): the fast tier's pre-tokenised prompts and the built-in demo cases with their references.
struct CascadeFile: Decodable {
    let marker_id: Int, audio_token_id: Int, audio_bos_id: Int, audio_eos_id: Int
    let scenarios: [FastScenario]
    let cases: [DemoCase]

    static func load() throws -> CascadeFile {
        try JSONDecoder().decode(CascadeFile.self, from: Data(contentsOf: FastEngine.url("cascade", "json")))
    }
}

/// An audio question the fast tier can ask about a recording (pre-tokenised; the app's scenario list).
struct FastScenario: Decodable, Identifiable, Hashable {
    let id: String, title: String, question: String, options: [String], descriptions: [String?], multi: Bool, rules: String?
    let fast: FastPrompt
}

/// A built-in case: our own test inputs (synthetic clips, demo examples), never benchmark items.
struct DemoCase: Decodable, Identifiable {
    let id: String, title: String, kind: Router.InputKind
    let media: String?, text: String?
    let question: String, options: [String], descriptions: [String?], multi: Bool, rules: String?
    let source: String
    let big_ref: [Double]?, big_ref_tokens: Int?
    let fast: FastPrompt?

    static func bundled(_ file: String) -> URL? {
        let url = URL(fileURLWithPath: file)
        return Bundle.main.url(forResource: url.deletingPathExtension().lastPathComponent, withExtension: url.pathExtension)
    }
}

/// One question for the cascade, whatever its source (built-in case, recording, typed text, picked photo).
struct Question: Sendable {
    let id: String, title: String, kind: Router.InputKind
    let question: String, options: [String], descriptions: [String?], multi: Bool, rules: String?
    var text: String? = nil
    var samples: [Float]? = nil
    var imageURL: URL? = nil
    var fast: FastPrompt? = nil
    var bigRef: [Double]? = nil, bigRefTokens: Int? = nil

    /// The Smaug-Decisions prompt (smaug_decide.py render of abacusai/Smaug-Decisions, with the same question, options and descriptions the fast
    /// tier saw, like the Python Cascade passing the same arguments to both models).
    var bigPrompt: String { bigPrompt(maxTokens: Prompt.maxTokens, imageTokens: Prompt.imageTokens) }

    /// The same at a smaller MAX_TOKENS (the on-device 2048-token cache) and the build's image token budget.
    func bigPrompt(maxTokens: Int, imageTokens: Int) -> String {
        let input: Prompt.Input
        switch kind {
        case .text: input = .text(text ?? "")
        case .image: input = .image
        case .audio: input = .audio(samples: min(samples?.count ?? 0, Prompt.maxAudioSeconds * Prompt.sampleRate))
        }
        return Prompt.decision(question: question, options: options, multi: multi, input: input, rules: rules, descriptions: descriptions,
                               maxTokens: maxTokens, imageTokens: imageTokens)
    }
}

/// What the cascade did with one question.
struct Outcome: Identifiable, Sendable {
    let id = UUID()
    let questionId: String, title: String, kind: Router.InputKind, question: String, options: [String], multi: Bool
    let route: Router.Route
    let tau: Double
    let fast: FastResult?, fastConfidence: Double?
    let big: BigResult?, bigLoadMs: Double?
    let memory: MemorySnapshot
    let note: String?
    let fastRef: [Double]?, bigRef: [Double]?
    var bigBuild: String? = nil   // "vis560, vision attached"
    /// Raw error text behind a short note (Details only).
    var detail: String? = nil

    /// The probabilities of the tier whose answer stands.
    var final: [Double]? { big?.probabilities ?? fast?.probabilities }
    var answeredBy: String {
        switch route {
        case .fast: return "0.7B audio model"
        case .escalated, .routed: return "Smaug-Decisions"
        case .wouldEscalate, .unsure: return "0.7B audio model (unsure)"
        case .needsBig, .notInstalled: return "—"
        }
    }
    var answer: String {
        guard let p = final, !p.isEmpty else { return "No answer on this device" }
        if multi {
            let sel = zip(options, p).filter { $0.1 > 0.5 }.map(\.0)
            return sel.isEmpty ? "None selected" : sel.joined(separator: ", ")
        }
        return options[p.indices.max { p[$0] < p[$1] }!]
    }
    var topP: Double? { final.map { multi ? 0 : ($0.max() ?? 0) } }
    var totalMs: Double { (fast?.msTotal ?? 0) + (big.map { $0.prefillMs + $0.readoutMs } ?? 0) }
}
