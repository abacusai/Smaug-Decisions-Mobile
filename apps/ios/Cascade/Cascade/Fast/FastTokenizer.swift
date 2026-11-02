import Foundation

/// Free-form fast-tier prompts without the audio model's tokenizer. Smaug-Audio-Decisions-0.7B and Smaug-Decisions share the Gemma 4 BPE tokenizer (same
/// 262,144-entry vocabulary and 514,906 merges; tools/tokenizer_check.py). The audio model differs only in its special handling: <unused0>
/// (id 6, also in Smaug-Decisions' vocabulary) is an added special token, the processor expands <|audio|> to <|audio> + N x <|audio|> +
/// <audio|>, and a prompt is wrapped in <bos> ... <eos>. Splitting the rendered prompt at those pieces and tokenizing the text in
/// between with Smaug-Decisions' tokenizer gives the audio model's ids exactly (checked on all pre-tokenised prompts).
enum FastTokenizer {
    static let bos = 2, eos = 1
    static let markerText = "<unused0>", audioText = "<|audio|>"

    /// smaug_decide.prompt.render_marker for an audio input.
    static func renderMarker(question: String, options: [String], descriptions: [String?], multi: Bool, rules: String?) -> String {
        let budget = charBudget(options: options.count)
        var pre = "task: decide | "
        if let rules, !rules.isEmpty { pre += "rules: \(Prompt.trim(rules, Prompt.floorDiv(budget, 2)))\n" }
        pre += "input: " + audioText
        var post = "\nquestion: \(question)" + (multi ? " (select all that apply)" : "") + "\n"
        for (opt, desc) in zip(options, descriptions) {
            post += "\(markerText) \(opt)" + ((desc ?? "").isEmpty ? "" : ": \(desc!)") + "\n"
        }
        return pre + post
    }

    /// char_budget('marker', 10 s * 25 + 2 audio tokens, k): the app's 10 s window, as make_scenarios.py renders it.
    static func charBudget(options k: Int) -> Int { Int(max(300, (1536.0 - 252 - 60 - 8 * Double(k)) * 3.5)) }

    /// (pre ids through <|audio>, post ids from <audio|> through <eos>) for a rendered marker prompt.
    static func ids(text: String, marker: Int, audioBOS: Int, audioEOS: Int, tokenize: (String) throws -> [Int]) throws -> ([Int], [Int]) {
        let halves = text.components(separatedBy: audioText)
        guard halves.count == 2 else { throw DeciderError(message: "prompt needs exactly one \(audioText)") }
        func enc(_ s: String) throws -> [Int] { s.isEmpty ? [] : try tokenize(s) }
        let pre = [bos] + (try enc(halves[0])) + [audioBOS]
        let parts = halves[1].components(separatedBy: markerText)
        var post = [audioEOS] + (try enc(parts[0]))
        for p in parts.dropFirst() { post += [marker] + (try enc(p)) }
        return (pre, post + [eos])
    }

    /// A fast-tier prompt for any question, tokenized on the phone.
    static func prompt(question: String, options: [String], descriptions: [String?], multi: Bool, rules: String?, marker: Int,
                       audioBOS: Int, audioEOS: Int, tokenize: (String) throws -> [Int]) throws -> FastPrompt {
        let text = renderMarker(question: question, options: options, descriptions: descriptions, multi: multi, rules: rules)
        let (pre, post) = try ids(text: text, marker: marker, audioBOS: audioBOS, audioEOS: audioEOS, tokenize: tokenize)
        let marks = post.indices.filter { post[$0] == marker }
        guard marks.count == options.count else { throw DeciderError(message: "an option text contains the marker") }
        return FastPrompt(pre_ids: pre, post_ids: post, marker_offsets_in_post: marks, text: text)
    }
}
