import XCTest

/// The cascade's own logic against the Python references (goldens/cascade_goldens.json from tools/make_goldens.py).
final class CascadeTests: XCTestCase {
    struct Goldens: Decodable {
        struct Conf: Decodable { let p: [Double]; let multi: Bool; let confidence: Double }
        struct BigCase: Decodable {
            let id: String, kind: String, samples: Int, question: String, options: [String], descriptions: [String?], multi: Bool
            let rules: String?, text: String?, prompt: String
        }
        struct Marker: Decodable {
            let id: String, question: String, options: [String], descriptions: [String?], multi: Bool, rules: String?, text: String
        }
        struct Device: Decodable {
            let id: String, kind: String, samples: Int, image_tokens: Int, question: String, options: [String], descriptions: [String?]
            let multi: Bool, rules: String?, text: String?, prompt: String
            let counts: [String: Int], fitted_max_tokens: Int, fitted_tokens: Int, fitted_prompt: String
        }
        let confidence: [Conf]
        let big_prompts: [BigCase]
        let marker_texts: [Marker]
        let device_budget: [Device]
    }

    /// The on-device budget (MAX_TOKENS 2000 for the 2048-token cache): the render matches smaug_decide's at that budget, and
    /// Prompt.fitted shrinks an overflowing prompt exactly as the Python replay does with the real Smaug-Decisions tokenizer counts (rules and
    /// text each take a share of the char budget, so both long overflows: 2486 -> 1846 tokens).
    func testDeviceBudgetMatchesPython() throws {
        let g = try goldens()
        XCTAssertGreaterThanOrEqual(g.device_budget.count, 6)
        var shrunk = 0
        for c in g.device_budget {
            var q = Question(id: c.id, title: c.id, kind: Router.InputKind(rawValue: c.kind)!, question: c.question, options: c.options,
                             descriptions: c.descriptions, multi: c.multi, rules: c.rules)
            q.text = c.text
            if c.kind == "audio" { q.samples = [Float](repeating: 0, count: c.samples) }
            XCTAssertEqual(Array(q.bigPrompt(maxTokens: Prompt.deviceMaxTokens, imageTokens: c.image_tokens).utf8), Array(c.prompt.utf8), c.id)
            let fit = try Prompt.fitted(limit: Prompt.deviceMaxTokens, render: { q.bigPrompt(maxTokens: $0, imageTokens: c.image_tokens) },
                                        count: { p in try XCTUnwrap(c.counts[p], "\(c.id): Swift rendered a prompt Python did not") })
            XCTAssertEqual(fit.maxTokens, c.fitted_max_tokens, c.id)
            XCTAssertEqual(fit.tokens, c.fitted_tokens, c.id)
            XCTAssertLessThanOrEqual(fit.tokens, Prompt.deviceMaxTokens, c.id)
            XCTAssertEqual(Array(fit.prompt.utf8), Array(c.fitted_prompt.utf8), c.id)
            if fit.maxTokens < Prompt.deviceMaxTokens { shrunk += 1 }
        }
        XCTAssertGreaterThanOrEqual(shrunk, 1)
        XCTAssertLessThan(Prompt.deviceMaxTokens, 2048)   // room for the scored answer token in the 2048-token cache
    }

    /// Router.confidence == smaug_decide.cascade.confidence.
    func testConfidenceMatchesPython() throws {
        let g = try goldens()
        XCTAssertGreaterThanOrEqual(g.confidence.count, 60)
        for c in g.confidence { XCTAssertEqual(Router.confidence(c.p, multi: c.multi), c.confidence, accuracy: 1e-12, "\(c.p) multi=\(c.multi)") }
    }

    func testRouting() {
        XCTAssertEqual(Router.route(kind: .audio, fastConfidence: 0.95, tau: 0.9, bigAvailable: true), .fast)
        XCTAssertEqual(Router.route(kind: .audio, fastConfidence: 0.9, tau: 0.9, bigAvailable: true), .fast)   // >= tau answers
        XCTAssertEqual(Router.route(kind: .audio, fastConfidence: 0.89, tau: 0.9, bigAvailable: true), .escalated)
        XCTAssertEqual(Router.route(kind: .audio, fastConfidence: 0.89, tau: 0.9, bigAvailable: false), .wouldEscalate)
        XCTAssertEqual(Router.route(kind: .audio, fastConfidence: 0.95, tau: 0.9, bigAvailable: false), .fast)
        XCTAssertEqual(Router.route(kind: .text, fastConfidence: nil, tau: 0.9, bigAvailable: true), .routed)
        XCTAssertEqual(Router.route(kind: .image, fastConfidence: nil, tau: 0.9, bigAvailable: true), .routed)
        XCTAssertEqual(Router.route(kind: .image, fastConfidence: nil, tau: 0.9, bigAvailable: false), .needsBig)
        XCTAssertTrue(Router.startsOnFast(.audio))
        XCTAssertFalse(Router.startsOnFast(.text))
        XCTAssertFalse(Router.startsOnFast(.image))
    }

    /// The Smaug-Decisions prompt of every built-in case and scenario is byte-identical to the Smaug-Decisions reference render.
    func testBigPromptsMatchReference() throws {
        let g = try goldens()
        XCTAssertGreaterThanOrEqual(g.big_prompts.count, 12)
        for c in g.big_prompts {
            var q = Question(id: c.id, title: c.id, kind: Router.InputKind(rawValue: c.kind)!, question: c.question, options: c.options,
                             descriptions: c.descriptions, multi: c.multi, rules: c.rules)
            q.text = c.text
            if c.kind == "audio" { q.samples = [Float](repeating: 0, count: c.samples) }
            XCTAssertEqual(Array(q.bigPrompt.utf8), Array(c.prompt.utf8), c.id)
        }
    }

    /// FastTokenizer.renderMarker == smaug_decide.prompt.render_marker (the text the pre-tokenised ids came from).
    func testMarkerTextsMatchPython() throws {
        let g = try goldens()
        XCTAssertGreaterThanOrEqual(g.marker_texts.count, 10)
        for m in g.marker_texts {
            let t = FastTokenizer.renderMarker(question: m.question, options: m.options, descriptions: m.descriptions, multi: m.multi, rules: m.rules)
            XCTAssertEqual(Array(t.utf8), Array(m.text.utf8), m.id)
        }
    }

    /// The segmentation around <|audio|> and <unused0>: specials inserted by id, text pieces tokenized separately.
    func testFastTokenizerSegments() throws {
        var pieces: [String] = []
        let tok: (String) throws -> [Int] = { pieces.append($0); return [1000 + pieces.count] }
        let p = try FastTokenizer.prompt(question: "Q?", options: ["a", "b"], descriptions: [nil, "d"], multi: false, rules: nil,
                                         marker: 6, audioBOS: 256000, audioEOS: 258883, tokenize: tok)
        XCTAssertEqual(pieces, ["task: decide | input: ", "\nquestion: Q?\n", " a\n", " b: d\n"])
        XCTAssertEqual(p.pre_ids, [2, 1001, 256000])
        XCTAssertEqual(p.post_ids, [258883, 1002, 6, 1003, 6, 1004, 1])
        XCTAssertEqual(p.marker_offsets_in_post, [2, 4])
    }

    func testDevicePolicy() {
        XCTAssertEqual(DevicePolicy.estimatedLimitMB(physicalMB: 12 * 1024), 6144)
        XCTAssertEqual(DevicePolicy.estimatedLimitMB(physicalMB: 7.5 * 1024), 6144)   // "8 GB" phones report a little under 8
        XCTAssertEqual(DevicePolicy.estimatedLimitMB(physicalMB: 5.6 * 1024), 3054)
        XCTAssertEqual(DevicePolicy.estimatedLimitMB(physicalMB: 3.7 * 1024), 2068)
        XCTAssertEqual(DevicePolicy.build(physicalMB: 12 * 1024), .vis560)
        XCTAssertEqual(DevicePolicy.build(physicalMB: 7.6 * 1024), .vis560)
        XCTAssertEqual(DevicePolicy.build(physicalMB: 5.6 * 1024), .vis280)
        XCTAssertNil(DevicePolicy.build(physicalMB: 3.7 * 1024))
        XCTAssertEqual(DevicePolicy.mode(build: .vis560, availableMB: 6144 - 100), .full)
        XCTAssertEqual(DevicePolicy.mode(build: .vis280, availableMB: 3054 - 100), .full)     // a 6 GB phone runs the cascade
        XCTAssertEqual(DevicePolicy.mode(build: nil, availableMB: 9999), .fastOnly)
        XCTAssertEqual(DevicePolicy.mode(build: .vis280, availableMB: DevicePolicy.Build.vis280.loadNeedMB - 1), .fastOnly)
        XCTAssertLessThan(DevicePolicy.Build.vis280.visionNeedMB, DevicePolicy.Build.vis560.visionNeedMB)
        XCTAssertLessThanOrEqual(DevicePolicy.Build.vis280.loadNeedMB + DevicePolicy.Build.vis280.visionNeedMB, 3054 - 100)
        let m = MemorySnapshot(footprintMB: 100, footprintPeakMB: 100, residentMB: 700, residentPeakMB: 700, availableMB: 5000)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m).mode, .full)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m).limitSource, "os_proc_available_memory")
        XCTAssertEqual(DevicePolicy.evaluate(memory: m, ramOverrideGB: 6).build, .vis280)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m, ramOverrideGB: 6).mode, .full)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m, ramOverrideGB: 8).build, .vis560)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m, ramOverrideGB: 4).mode, .fastOnly)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m, forcedMode: .fastOnly).mode, .fastOnly)
        XCTAssertEqual(DevicePolicy.evaluate(memory: m, ramOverrideGB: 6, forcedBuild: .vis560).build, .vis560)
        let low = MemorySnapshot(footprintMB: 100, footprintPeakMB: 100, residentMB: 700, residentPeakMB: 700, availableMB: 1000)
        XCTAssertEqual(DevicePolicy.evaluate(memory: low).mode, .fastOnly)
    }

    private func goldens() throws -> Goldens {
        let url = try XCTUnwrap(Bundle(for: CascadeTests.self).url(forResource: "cascade_goldens", withExtension: "json"))
        return try JSONDecoder().decode(Goldens.self, from: Data(contentsOf: url))
    }
}
