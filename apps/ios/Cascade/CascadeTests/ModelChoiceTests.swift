import XCTest

/// Settings › Models: routing per choice, variant selection from the manifest, the switch plan, and the int8 embedding row.
final class ModelChoiceTests: XCTestCase {
    /// Every choice x input kind x fast-tier confidence x Smaug-Decisions availability.
    func testRoutingPerChoice() {
        typealias R = Router.Route
        let hi = 0.95, lo = 0.6, tau = 0.9
        // (choice, audio confident, audio unsure, audio unsure without Smaug-Decisions loaded, text, image, text without Smaug-Decisions loaded)
        let table: [(ModelChoice, R, R, R, R, R, R)] = [
            (.full, .fast, .escalated, .wouldEscalate, .routed, .routed, .needsBig),
            (.audioText, .fast, .escalated, .wouldEscalate, .routed, .notInstalled, .needsBig),
            (.textImage, .fast, .unsure, .unsure, .routed, .routed, .needsBig),
            (.audioOnly, .fast, .unsure, .unsure, .notInstalled, .notInstalled, .notInstalled),
        ]
        for (c, aHi, aLo, aLoNoBig, text, image, textNoBig) in table {
            XCTAssertEqual(Router.route(kind: .audio, fastConfidence: hi, tau: tau, bigAvailable: true, choice: c), aHi, "\(c) audio confident")
            XCTAssertEqual(Router.route(kind: .audio, fastConfidence: lo, tau: tau, bigAvailable: true, choice: c), aLo, "\(c) audio unsure")
            XCTAssertEqual(Router.route(kind: .audio, fastConfidence: lo, tau: tau, bigAvailable: false, choice: c), aLoNoBig, "\(c) audio unsure, no Smaug-Decisions")
            XCTAssertEqual(Router.route(kind: .text, fastConfidence: nil, tau: tau, bigAvailable: true, choice: c), text, "\(c) text")
            XCTAssertEqual(Router.route(kind: .image, fastConfidence: nil, tau: tau, bigAvailable: true, choice: c), image, "\(c) image")
            XCTAssertEqual(Router.route(kind: .text, fastConfidence: nil, tau: tau, bigAvailable: false, choice: c), textNoBig, "\(c) text, no Smaug-Decisions")
        }
        // the default choice keeps the original rule
        XCTAssertEqual(Router.route(kind: .audio, fastConfidence: lo, tau: tau, bigAvailable: true), .escalated)
    }

    func testOfferedChoicesFollowDevicePolicy() {
        XCTAssertEqual(ModelChoice.offered(mode: .fastOnly), [.audioOnly])
        XCTAssertEqual(ModelChoice.offered(mode: .full), ModelChoice.allCases)
        XCTAssertEqual(ModelChoice.defaultChoice(mode: .full), .full)
        let d = UserDefaults(suiteName: "ModelChoiceTests")!
        d.set(ModelChoice.textImage.rawValue, forKey: ModelChoice.storeKey)
        XCTAssertEqual(ModelChoice.current(mode: .full, arguments: [], defaults: d), .textImage)
        XCTAssertEqual(ModelChoice.current(mode: .fastOnly, arguments: [], defaults: d), .audioOnly)   // never Smaug-Decisions on Fast only
        XCTAssertEqual(ModelChoice.current(mode: .full, arguments: ["--models", "audio_text"], defaults: d), .audioText)
        d.removePersistentDomain(forName: "ModelChoiceTests")
    }

    /// The first-run setup screen: once, only while no Smaug-Decisions file is here, never in automation launches.
    func testFirstRunShowsOnceBeforeAnyBigFile() {
        XCTAssertTrue(FirstRun.shouldShow(seen: false, bigOnDevice: false, downloadPending: false, arguments: []))
        XCTAssertFalse(FirstRun.shouldShow(seen: true, bigOnDevice: false, downloadPending: false, arguments: []))
        XCTAssertFalse(FirstRun.shouldShow(seen: false, bigOnDevice: true, downloadPending: false, arguments: []))
        XCTAssertFalse(FirstRun.shouldShow(seen: false, bigOnDevice: false, downloadPending: true, arguments: []))
        XCTAssertFalse(FirstRun.shouldShow(seen: false, bigOnDevice: false, downloadPending: false, arguments: ["--autorun", "a1"]))
        XCTAssertTrue(FirstRun.shouldShow(seen: true, bigOnDevice: true, downloadPending: false, arguments: ["--first-run", "--snapshots", "x@3"]))
        XCTAssertEqual(FirstRun.recommended(mode: .full), .full)
        XCTAssertEqual(FirstRun.recommended(mode: .fastOnly), .audioOnly)
        XCTAssertTrue(ModelChoice.offered(mode: .fastOnly).contains(FirstRun.recommended(mode: .fastOnly)))
    }

    private func manifest(withAudioText: Bool = true) throws -> ModelManifest {
        var v: [[String: Any]] = [
            ["variant": "full_vis560", "file": "f560.litertlm", "bytes": 4_600, "sha256": "a", "modalities": ["text", "image", "audio"], "vision_tokens": 560],
            ["variant": "full_vis280", "file": "f280.litertlm", "bytes": 4_600, "sha256": "b", "modalities": ["text", "image", "audio"], "vision_tokens": 280],
            ["variant": "text_image_vis560", "file": "ti560.litertlm", "bytes": 4_300, "sha256": "c", "modalities": ["text", "image"], "vision_tokens": 560],
            ["variant": "text_image_vis280", "file": "ti280.litertlm", "bytes": 4_300, "sha256": "d", "modalities": ["text", "image"], "vision_tokens": 280],
        ]
        if withAudioText { v.append(["variant": "audio_text", "file": "at.litertlm", "bytes": 4_400, "sha256": "e", "modalities": ["text", "audio"], "vision_tokens": NSNull()]) }
        let json: [String: Any] = ["version": 1, "base_url": "https://example.com/", "estimated": ["audio_text"], "variants": v]
        return try ModelManifest.decode(JSONSerialization.data(withJSONObject: json), source: "test")
    }

    func testVariantPerChoice() throws {
        let m = try manifest()
        XCTAssertNil(m.variant(for: .audioOnly, visionTokens: 280))
        XCTAssertEqual(m.variant(for: .full, visionTokens: 280)?.variant, "full_vis280")
        XCTAssertEqual(m.variant(for: .full, visionTokens: 560)?.variant, "full_vis560")
        XCTAssertEqual(m.variant(for: .textImage, visionTokens: 280)?.variant, "text_image_vis280")
        XCTAssertEqual(m.variant(for: .audioText, visionTokens: 280)?.variant, "audio_text")
        XCTAssertTrue(m.isEstimated(m.variant(for: .audioText, visionTokens: 280)!))
        XCTAssertEqual(m.url(m.variants[0])?.absoluteString, "https://example.com/f560.litertlm")
        // no audio+text variant yet: the smallest full build covers it
        XCTAssertEqual(try manifest(withAudioText: false).variant(for: .audioText, visionTokens: 280)?.modalitySet, [.text, .image, .audio])
    }

    func testSwitchPlan() throws {
        let m = try manifest()
        let have: Set<String> = ["f280.litertlm"]
        let keep = ModelPlan.make(choice: .full, manifest: m, visionTokens: 280, installed: have.contains)
        XCTAssertNil(keep.download); XCTAssertTrue(keep.delete.isEmpty)
        let smaller = ModelPlan.make(choice: .audioText, manifest: m, visionTokens: 280, installed: have.contains)
        XCTAssertEqual(smaller.download?.variant, "audio_text")
        XCTAssertEqual(smaller.delete.map(\.file), ["f280.litertlm"])   // offered for deletion, deleted only after confirmation
        let audioOnly = ModelPlan.make(choice: .audioOnly, manifest: m, visionTokens: 280, installed: have.contains)
        XCTAssertNil(audioOnly.target); XCTAssertNil(audioOnly.download)
        XCTAssertEqual(audioOnly.freedBytes, 4_600)
    }

    /// The bundled manifest: every variant has exactly the quant build's fields, and every Smaug-Decisions choice resolves at 280 and 560.
    func testBundledManifest() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "models_manifest", withExtension: "json"))
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        for v in try XCTUnwrap(raw["variants"] as? [[String: Any]]) {
            XCTAssertEqual(Set(v.keys).subtracting(["url"]), ["variant", "file", "bytes", "sha256", "modalities", "vision_tokens"])
        }
        let m = try ModelManifest.decode(Data(contentsOf: url), source: "bundled")
        for c in ModelChoice.allCases where c.usesBig { for vt in [280, 560] { XCTAssertNotNil(m.variant(for: c, visionTokens: vt), "\(c) @\(vt)") } }
        for v in m.variants where !m.isEstimated(v) { XCTAssertEqual(v.sha256.count, 64, v.variant) }
        // the Smaug-Decisions-LiteRT builds: full / no-audio / no-vision resolve to their own variants
        XCTAssertEqual(m.variant(for: .full, visionTokens: 280)?.variant, "full_vis280")
        XCTAssertEqual(m.variant(for: .textImage, visionTokens: 280)?.variant, "noaudio_vis280")
        let full = try XCTUnwrap(m.variant(for: .full, visionTokens: 280))
        XCTAssertEqual(m.url(full)?.absoluteString, "https://huggingface.co/abacusai/Smaug-Decisions-LiteRT/resolve/main/smaug-decisions-full-vis280.litertlm")
        XCTAssertEqual(m.variant(for: .audioText, visionTokens: 560)?.variant, "novision")
        // decoder cache estimate vs the measured 2,044,028,752 B for every Smaug-Decisions-LiteRT variant
        for v in m.variants { XCTAssertEqual(Double(v.cacheBytesWithoutVision - (v.modalitySet.contains(.audio) ? 300_000_000 : 0)), 2.044e9, accuracy: 0.05e9, v.variant) }
    }

    func testInt8EmbeddingRowDequantises() throws {
        let d = EmbeddingTable.dim
        var q = [Int8](repeating: 0, count: 2 * d)
        for j in 0..<d { q[j] = Int8(j % 255 - 127); q[d + j] = Int8(127 - j % 255) }
        let scales: [Float] = [0.01, 0.5]
        let t = try EmbeddingTable(int8: q.withUnsafeBufferPointer { Data(buffer: $0) }, scales: scales.withUnsafeBufferPointer { Data(buffer: $0) })
        XCTAssertEqual(t.vocab, 2)
        var out = [Float](repeating: 0, count: d)
        for r in 0..<2 {
            out.withUnsafeMutableBufferPointer { t.row(r, into: $0.baseAddress!) }
            for j in 0..<d { XCTAssertEqual(out[j], Float(q[r * d + j]) * scales[r], accuracy: 1e-6) }
        }
    }
}
