import Foundation

/// The big tier: Smaug-Decisions on LiteRT-LM (CPU), adapted from an earlier single-model LiteRT-LM demo. One engine while
/// loaded (the cascade creates it on the first escalation and can release it under memory pressure), a fresh session per question.
/// Use from one serial queue only: the patched runtime's logits readout is process-global.
final class BigEngine: @unchecked Sendable {
    /// The Smaug-Decisions-LiteRT builds have a 2048-token KV cache: the engine's max tokens must match, and prompts
    /// are rendered with Prompt.deviceMaxTokens so they fit.
    static let maxTokens = 2048

    /// Image soft tokens of the loaded build: 560 (vis560, as trained) or 280 (vis280, the 6 GB tier).
    let visualTokenBudget: Int
    /// Whether the vision encoder is attached (the cascade attaches it only when an image arrives).
    let hasVision: Bool
    let modelPath: String

    /// The linked CLiteRTLM has our harness patch: media prefill + absolute letter logits (one scoring step per question).
    let patched: Bool
    private let engine: UnsafeMutableRawPointer
    private let letterIds: [Int32]

    var runtimeDescription: String {
        patched ? "patched runtime: text, image, audio; one-answer and select-all"
            : "stock runtime: text only, one-answer (one prefill per option)"
    }

    init(modelPath: String, threads: Int, vision: Bool, visualTokens: Int, backend: String = "cpu") throws {
        let patched = decide_has_harness() == 1
        // The encoders are only created when the runtime can feed them (patched build); vision only when asked for.
        guard let engine = decide_engine_create(modelPath, backend, try Self.cacheDir(), Int32(Self.maxTokens), Int32(threads), patched && vision,
                                                patched, decide_c_api_v1() == 1) else {
            throw DeciderError(message: "Could not load the model: \(Self.lastError())")
        }
        var ids: [Int32] = []
        for letter in Prompt.letters {
            let id = decide_single_token(engine, String(letter))
            guard id >= 0 else {
                decide_engine_delete(engine)
                throw DeciderError(message: "Letter \(letter) is not a single token: \(Self.lastError())")
            }
            ids.append(id)
        }
        self.patched = patched
        self.engine = engine
        self.letterIds = ids
        self.hasVision = patched && vision
        self.visualTokenBudget = visualTokens
        self.modelPath = modelPath
    }

    deinit { decide_engine_delete(engine) }

    /// Token ids of `text` with the model's own tokenizer (no BOS), for the fast tier's free-form prompts.
    func tokenize(_ text: String) throws -> [Int] {
        let n = Int(decide_tokenize(engine, text, nil, 0))
        guard n >= 0 else { throw DeciderError(message: "Tokenize failed: \(Self.lastError())") }
        var ids = [Int32](repeating: 0, count: max(n, 1))
        let m = ids.withUnsafeMutableBufferPointer { Int(decide_tokenize(engine, text, $0.baseAddress, n)) }
        guard m == n else { throw DeciderError(message: "Tokenize failed: \(Self.lastError())") }
        return ids.prefix(n).map { Int($0) }
    }

    /// `prompt` is the rendered prompt (with its leading <bos>, which is stripped here). Returns one probability per option.
    func decide(prompt: String, options n: Int, multi: Bool, media: MediaFile?) throws -> BigResult {
        guard (2...Prompt.letters.count).contains(n) else { throw DeciderError(message: "Give 2 to 20 options.") }
        let text = Prompt.stripBOS(prompt)
        return try patched ? decideWithLogits(text, n, multi, media) : decideStock(text, n, multi, media)
    }

    /// One prefill, one scoring step, then the raw logits of every option letter at that step.
    private func decideWithLogits(_ text: String, _ n: Int, _ multi: Bool, _ media: MediaFile?) throws -> BigResult {
        let json = try mediaJSON(media)
        let session = try newSession()
        defer { decide_session_delete(session) }
        let t0 = DispatchTime.now()
        try check(decide_prefill_multimodal(session, engine, text, json, Int32(visualTokenBudget)), "Prefill")
        let t1 = DispatchTime.now()
        let tokens = Int(decide_last_prefill_tokens(session))   // before scoring, which adds a turn
        var logp: Float = 0
        try check(decide_score_text(session, String(Prompt.letters[0]), &logp), "Scoring")
        var z = [Float](repeating: 0, count: n)
        var logsumexp: Float = 0
        try check(decide_last_logits(letterIds, n, &z, &logsumexp), "Logits readout")
        let t2 = DispatchTime.now()
        let capped = Readout.capped(z.map { Double($0) })
        return BigResult(probabilities: Readout.probabilities(capped, multi: multi), logits: capped,
                              promptTokens: tokens >= 0 ? tokens : estimateTokens(text), tokensFromRuntime: tokens >= 0,
                              prefillMs: Self.ms(t0, t1), readoutMs: Self.ms(t1, t2), patched: true)
    }

    /// Fallback for the stock prebuilt runtime: log p(letter) per option, each on a fresh session (it has no session clone,
    /// and scoring one session twice is corrupted, LiteRT-LM #3561). softmax(log p) renormalizes over the options. It equals
    /// the softmax of the RAW logits: the softcap cannot be applied without absolute logits, so the top answer matches the
    /// reference but the probabilities come out sharper.
    private func decideStock(_ text: String, _ n: Int, _ multi: Bool, _ media: MediaFile?) throws -> BigResult {
        if media != nil {
            throw DeciderError(message: "Images and audio need the patched runtime (build_runtime_ios.sh without --prebuilt).")
        }
        if multi { throw DeciderError(message: "Select-all needs absolute letter logits, i.e. the patched runtime.") }
        var logp: [Double] = []
        var prefillMs = 0.0, readoutMs = 0.0
        for k in 0..<n {
            let session = try newSession()
            defer { decide_session_delete(session) }
            let t0 = DispatchTime.now()
            try check(decide_prefill_text(session, text), "Prefill")
            let t1 = DispatchTime.now()
            var score: Float = 0
            try check(decide_score_text(session, String(Prompt.letters[k]), &score), "Scoring")
            prefillMs += Self.ms(t0, t1)
            readoutMs += Self.ms(t1, DispatchTime.now())
            logp.append(Double(score))
        }
        return BigResult(probabilities: Readout.softmax(logp), logits: nil, promptTokens: estimateTokens(text),
                              tokensFromRuntime: false, prefillMs: prefillMs, readoutMs: readoutMs, patched: false)
    }

    private func newSession() throws -> UnsafeMutableRawPointer {
        guard let session = decide_session_create(engine) else {
            throw DeciderError(message: "Could not create a session: \(Self.lastError())")
        }
        return session
    }

    /// Tokens of the rendered prompt as the runtime will prefill them: text tokens + BOS, with the media placeholder replaced by
    /// an estimate of its encoded length (audio 25/s + 2; image: the build's budget plus 10% headroom, LiteRT-LM encoded the
    /// 800x300 sign at 560 as 615 - 112 text tokens). Used to keep prompts inside the 2048-token KV cache.
    func promptTokens(_ prompt: String, audioSamples: Int?, image: Bool) throws -> Int {
        let text = Prompt.stripBOS(prompt)
        let n = Int(decide_count_tokens(engine, text))
        guard n >= 0 else { throw DeciderError(message: "Token count failed: \(Self.lastError())") }
        var media = 0.0
        if let audioSamples { media = Prompt.mediaTokens(audioSamples: audioSamples, image: false) - 1 }
        if image { media = Double(visualTokenBudget) * 1.1 + 2 - 1 }
        return n + 1 + Int(media.rounded(.up))
    }

    /// Text tokens + BOS (media tokens not included).
    private func estimateTokens(_ text: String) -> Int {
        let n = Int(decide_count_tokens(engine, text))
        return n >= 0 ? n + 1 : -1
    }

    private func mediaJSON(_ media: MediaFile?) throws -> String {
        guard let media else { return "" }
        let data = try JSONSerialization.data(withJSONObject: [["type": media.kind.rawValue, "path": media.path]])
        return String(decoding: data, as: UTF8.self)
    }

    private func check(_ rc: Int32, _ what: String) throws {
        if rc != 0 { throw DeciderError(message: "\(what) failed: \(Self.lastError())") }
    }

    private static func lastError() -> String { String(cString: decide_last_error()) }

    private static func ms(_ a: DispatchTime, _ b: DispatchTime) -> Double {
        Double(b.uptimeNanoseconds - a.uptimeNanoseconds) / 1e6
    }

    /// LiteRT-LM's XNNPack weight cache (about 2.8 GB, written on the first load). It must be writable and kept: without it the
    /// engine's dirty footprint is about 4.8 GB instead of 0.8 GB (quant.md). Application Support is never purged by iOS (Caches
    /// can be, under storage pressure); the folder is excluded from iCloud / device backups since it is rebuilt from the model.
    static func cacheDir() throws -> String {
        var dir = URL.applicationSupportDirectory.appendingPathComponent("litert-cache", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        try dir.setResourceValues(values)
        return dir.path
    }

    /// Bytes of the weight-cache files LiteRT-LM wrote for one model file (they are named "<file>_<mtime>_<size>.<part>.<kind>").
    static func cacheBytes(model file: String) -> Int64 {
        guard let dir = try? cacheDir(), let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { return 0 }
        return names.filter { $0.hasPrefix(file + "_") }.reduce(Int64(0)) {
            $0 + Int64((try? FileManager.default.attributesOfItem(atPath: dir + "/" + $1)[.size] as? Int) ?? 0)
        }
    }

    /// Bytes of cache files that name no .litertlm file (e.g. a backend's program cache with another naming scheme).
    static func unattributedCacheBytes() -> Int64 {
        guard let dir = try? cacheDir(), let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { return 0 }
        return names.filter { !$0.contains(".litertlm_") }.reduce(Int64(0)) {
            $0 + Int64((try? FileManager.default.attributesOfItem(atPath: dir + "/" + $1)[.size] as? Int) ?? 0)
        }
    }

    /// Deletes the cache files of model files that are no longer in Documents (a deleted or replaced variant), returns bytes freed.
    @discardableResult
    static func dropStaleCaches(keeping present: Set<String>) -> Int64 {
        guard let dir = try? cacheDir(), let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { return 0 }
        var freed: Int64 = 0
        for n in names {
            guard let r = n.range(of: ".litertlm_") else { continue }   // only files we can attribute to a model file
            let model = String(n[..<r.lowerBound]) + ".litertlm"
            // LiteRT-LM names a cache "<file>_<mtime>_<size>.<part>.xnnpack_cache" (CPU) or "<file>_<mtime>_<size>_mldrift_*.bin"
            // (GPU); a cache for an older copy of a present file is stale too
            if present.contains(model), let token = fileToken(URL.documentsDirectory.appendingPathComponent(model)),
               n[r.upperBound...].hasPrefix(token + ".") || n[r.upperBound...].hasPrefix(token + "_") { continue }
            freed += Int64((try? FileManager.default.attributesOfItem(atPath: dir + "/" + n)[.size] as? Int) ?? 0)
            try? FileManager.default.removeItem(atPath: dir + "/" + n)
        }
        return freed
    }

    /// "<mtime>_<size>" of a model file, as LiteRT-LM puts it in its cache names.
    static func fileToken(_ url: URL) -> String? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: url.path), let d = a[.modificationDate] as? Date,
              let s = (a[.size] as? NSNumber)?.int64Value else { return nil }
        return "\(Int64(d.timeIntervalSince1970))_\(s)"
    }

    /// Size of the weight cache on disk, in MB.
    static func cacheMB() -> Double {
        guard let dir = try? cacheDir(), let files = FileManager.default.enumerator(atPath: dir) else { return 0 }
        var total = 0
        for case let f as String in files { total += (try? FileManager.default.attributesOfItem(atPath: dir + "/" + f)[.size] as? Int) ?? 0 }
        return Double(total) / 1_048_576
    }
}
