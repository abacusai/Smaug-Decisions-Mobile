import XCTest

/// Prompt.render / trim / char budget and the readout vs the Python reference (goldens from tools/make_goldens.py).
final class PromptTests: XCTestCase {
    struct RenderCase: Decodable {
        let question: String
        let options: [String]
        let media: String
        let text: String?
        let descriptions: [String?]?
        let multi: Bool
        let rules: String?
        let context: String?
        let audio_samples: Int
        let note: String
        let budget: Int
        let prompt: String
    }

    struct TrimCase: Decodable {
        let s: String
        let n: Int
        let out: String
    }

    struct ReadoutGoldens: Decodable {
        struct Case: Decodable {
            let z: [Double]
            let softmax: [Double]
            let sigmoid: [Double]
        }

        let b: Double
        let cases: [Case]
    }

    /// Smaug-Decisions constants (MAX_TOKENS 3072, IMAGE_TOKENS 560): the budget and the prompt must both match.
    func testRenderMatchesPython() throws {
        let cases = try load([RenderCase].self, "render")
        XCTAssertGreaterThanOrEqual(cases.count, 40)
        for c in cases {
            let samples = c.media == "audio" ? c.audio_samples : nil
            let budget = Prompt.charBudget(mediaTokens: Prompt.mediaTokens(audioSamples: samples, image: c.media == "image"),
                                           options: c.options.count)
            XCTAssertEqual(budget, c.budget, "budget: \(c.note)")
            assertBytesEqual(render(c, budget: budget), c.prompt, c.note)
        }
    }

    /// The Android goldens (older, smaller budgets, so the trimming paths inside render are exercised), at their budgets.
    func testRenderAtFixedBudgets() throws {
        let cases = try load([RenderCase].self, "render_fixed_budget")
        XCTAssertGreaterThanOrEqual(cases.count, 40)
        for c in cases { assertBytesEqual(render(c, budget: c.budget), c.prompt, c.note) }
    }

    func testTrimMatchesPython() throws {
        let cases = try load([TrimCase].self, "trim")
        XCTAssertGreaterThanOrEqual(cases.count, 80)
        for c in cases { assertBytesEqual(Prompt.trim(c.s, c.n), c.out, "n=\(c.n) s=\(c.s.prefix(40))") }
    }

    func testReadoutMatchesNumpy() throws {
        let g = try load(ReadoutGoldens.self, "readout")
        XCTAssertGreaterThanOrEqual(g.cases.count, 100)
        for c in g.cases {
            for (a, b) in zip(Readout.softmax(c.z), c.softmax) { XCTAssertEqual(a, b, accuracy: 1e-12) }
            for (a, b) in zip(Readout.sigmoid(c.z, b: g.b), c.sigmoid) { XCTAssertEqual(a, b, accuracy: 1e-12) }
        }
    }

    func testSoftcap() {
        let z = Readout.capped([0, 30, -1000, 45.5])
        XCTAssertEqual(z[0], 0)
        XCTAssertEqual(z[1], 30 * tanh(1), accuracy: 1e-12)
        XCTAssertEqual(z[2], -30, accuracy: 1e-9)
        XCTAssertLessThan(z[3], 30)
        XCTAssertEqual(Readout.multiBias, 8.010554313659668)
    }

    private func render(_ c: RenderCase, budget: Int) -> String {
        let tag: String? = c.media == "audio" ? Prompt.audioTag : c.media == "image" ? Prompt.imageTag : nil
        let descriptions = c.descriptions ?? Array(repeating: nil, count: c.options.count)
        return Prompt.render(question: c.question, options: c.options, descriptions: descriptions, multi: c.multi,
                             mediaTag: tag, rules: c.rules, context: c.context, textInput: tag == nil ? c.text : nil,
                             charBudget: budget)
    }

    /// Swift's String == compares canonically equivalent strings as equal, so compare the UTF-8 bytes too.
    private func assertBytesEqual(_ got: String, _ want: String, _ note: String, line: UInt = #line) {
        XCTAssertEqual(got, want, note, line: line)
        XCTAssertEqual(Array(got.utf8), Array(want.utf8), "bytes: \(note)", line: line)
    }

    private func load<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
        let url = try XCTUnwrap(Bundle(for: PromptTests.self).url(forResource: name, withExtension: "json"), "missing \(name).json")
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }
}
