import CoreML
import Foundation

/// The fast tier, Smaug-Audio-Decisions-0.7B on Core ML: adapted from the earlier fast-tier demo app, audio only, single audio unit.
/// Recorded audio -> log-mel (Swift) -> Core ML audio unit -> splice into the token embeddings of the pre-tokenised prompt ->
/// Core ML trunk -> linear head at the option markers -> softmax (one answer) or a sigmoid per option (select-all).
final class FastEngine: @unchecked Sendable {
    /// Phone: trunk on the Neural Engine when it can; the audio tower's graph does not compile for the Neural Engine, so GPU
    /// (389 ms on an iPhone 14 Pro Max). Simulator: CPU only. With GPU compute units the Simulator's
    /// Core ML returned all-zero outputs for both units (Xcode 26.6, iOS 26.5 Simulator), so the probabilities came out uniform.
    #if targetEnvironment(simulator)
    static var trunkUnits: MLComputeUnits { override ?? .cpuOnly }
    static var audioUnits: MLComputeUnits { override ?? .cpuOnly }
    #else
    static var trunkUnits: MLComputeUnits { override ?? .all }
    static var audioUnits: MLComputeUnits { override ?? .cpuAndGPU }
    #endif
    /// `--fast-units cpu|gpu|all` (launch argument) for experiments.
    static let override: MLComputeUnits? = {
        let a = ProcessInfo.processInfo.arguments
        guard let i = a.firstIndex(of: "--fast-units"), i + 1 < a.count else { return nil }
        return ["cpu": .cpuOnly, "gpu": .cpuAndGPU, "all": .all, "ane": .cpuAndNeuralEngine][a[i + 1]]
    }()
    static let audioSeconds = 10.0

    let markerId: Int, audioTokenId: Int
    private let trunk: MLModel, audio: MLModel
    /// The token-embedding table (int8 + per-row scales in the shipped bundle), memory-mapped.
    let table: EmbeddingTable
    private let dim = EmbeddingTable.dim, L: Int, audioFrames: Int, audioRows: Int
    private let headW: [Float], headB: Float
    private let mel = MelFrontEnd()
    private let maskNeg: Float = -30000

    static func url(_ name: String, _ ext: String) throws -> URL {
        guard let u = Bundle.main.url(forResource: name, withExtension: ext) else { throw DeciderError(message: "missing resource \(name).\(ext)") }
        return u
    }

    init(markerId: Int, audioTokenId: Int, progress: @escaping @Sendable (String) -> Void) throws {
        self.markerId = markerId; self.audioTokenId = audioTokenId
        let head = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.url("head", "json"))) as! [String: Any]
        headW = (head["weight"] as! [NSNumber]).map { $0.floatValue }; headB = (head["bias"] as! NSNumber).floatValue
        table = try EmbeddingTable.bundled()
        let tcfg = MLModelConfiguration(); tcfg.computeUnits = Self.trunkUnits
        progress("loading the 0.7B audio model (first launch compiles)…")
        trunk = try Self.model("trunk_L512_int8", tcfg)
        let acfg = MLModelConfiguration(); acfg.computeUnits = Self.audioUnits
        progress("loading the audio encoder…")
        audio = try Self.model("audio_10s_int8", acfg)
        L = trunk.modelDescription.inputDescriptionsByName["inputs_embeds"]?.multiArrayConstraint?.shape[1].intValue ?? 512
        audioFrames = audio.modelDescription.inputDescriptionsByName["features"]?.multiArrayConstraint?.shape[1].intValue ?? 999
        if let c = audio.modelDescription.outputDescriptionsByName["out"]?.multiArrayConstraint, c.shape.count == 3 { audioRows = c.shape[1].intValue }
        else { var t = audioFrames; for _ in 0..<2 { t = (t + 2 - 3) / 2 + 1 }; audioRows = t }
    }

    private static func model(_ name: String, _ cfg: MLModelConfiguration) throws -> MLModel {
        if let c = Bundle.main.url(forResource: name, withExtension: "mlmodelc") { return try MLModel(contentsOf: c, configuration: cfg) }
        let compiled = try MLModel.compileModel(at: try url(name, "mlpackage"))   // raw package in the bundle: compile once here
        return try MLModel(contentsOf: compiled, configuration: cfg)
    }

    /// A Core ML output as Float32 in logical (row-major) order, through withUnsafeBytes (valid for any backing, unlike the
    /// deprecated dataPointer), honouring strides.
    static func floats(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        a.withUnsafeBytes { raw in
            func at(_ flat: Int) -> Int {   // element offset of logical index `flat`
                var rem = flat, off = 0
                for d in stride(from: shape.count - 1, through: 0, by: -1) { off += (rem % shape[d]) * strides[d]; rem /= shape[d] }
                return off
            }
            let contiguous = strides == shape.indices.map { d in shape[(d + 1)...].reduce(1, *) }
            switch a.dataType {
            case .float16:
                let p = raw.bindMemory(to: Float16.self)
                for i in 0..<count { out[i] = Float(p[contiguous ? i : at(i)]) }
            case .float32:
                let p = raw.bindMemory(to: Float.self)
                for i in 0..<count { out[i] = p[contiguous ? i : at(i)] }
            default:
                let p = raw.bindMemory(to: Double.self)
                for i in 0..<count { out[i] = Float(p[contiguous ? i : at(i)]) }
            }
        }
        return out
    }

    /// Audio tokens for n valid 16 kHz samples: mel frames through the two stride-2 convolutions (kernel 3, padding 1), cap 750.
    static func audioTokens(samples n: Int) -> Int {
        var t = MelFrontEnd.frameCount(samples: n)
        for _ in 0..<2 { t = (t + 2 - 3) / 2 + 1 }
        return max(0, min(t, 750))
    }

    /// One decision: `samples` are 16 kHz mono; the last 10 s are used (the decision is about how the clip ends). `table`
    /// overrides the bundled embedding table (the int8 vs fp16 parity check).
    func run(_ p: FastPrompt, options k: Int, multi: Bool, samples: [Float], table override: EmbeddingTable? = nil) throws -> FastResult {
        let table = override ?? self.table
        let t0 = Date()
        let clip = Array(samples.suffix(Int(Self.audioSeconds * 16000)))
        guard !clip.isEmpty else { throw DeciderError(message: "empty audio") }
        guard p.marker_offsets_in_post.count == k else { throw DeciderError(message: "prompt has \(p.marker_offsets_in_post.count) markers for \(k) options") }
        let nAudio = Self.audioTokens(samples: clip.count)
        let (feats, mask) = mel.features(samples: clip, frames: audioFrames)
        let fa = try MLMultiArray(shape: [1, NSNumber(value: audioFrames), 128], dataType: .float32)
        let ma = try MLMultiArray(shape: [1, NSNumber(value: audioFrames)], dataType: .float32)
        feats.withUnsafeBufferPointer { _ = memcpy(fa.dataPointer, $0.baseAddress!, feats.count * 4) }
        mask.withUnsafeBufferPointer { _ = memcpy(ma.dataPointer, $0.baseAddress!, mask.count * 4) }
        let ta = Date()
        let audioEmb = try audio.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": fa, "mask": ma])).featureValue(for: "out")!.multiArrayValue!
        let msAudio = Date().timeIntervalSince(ta) * 1000
        let audioRowsF = Self.floats(audioEmb)   // [audioRows * dim], logical order
        let ids = p.pre_ids + Array(repeating: audioTokenId, count: nAudio) + p.post_ids
        let markers = p.marker_offsets_in_post.map { p.pre_ids.count + nAudio + $0 }
        guard ids.count <= L, let lastMark = markers.last, lastMark < L else { throw DeciderError(message: "prompt too long (\(ids.count) > \(L))") }
        let n = ids.count
        let xe = try MLMultiArray(shape: [1, NSNumber(value: L), NSNumber(value: dim)], dataType: .float32)
        let xp = xe.dataPointer.assumingMemoryBound(to: Float.self); xp.initialize(repeating: 0, count: L * dim)
        var audioRow = 0
        for (i, id) in ids.enumerated() {
            if id == audioTokenId, audioRow < min(nAudio, audioRows) {
                for j in 0..<dim { xp[i * dim + j] = audioRowsF[audioRow * dim + j] }
                audioRow += 1
            } else { table.row(id, into: xp + i * dim) }
        }
        let mk = try MLMultiArray(shape: [1, 1, NSNumber(value: L), NSNumber(value: L)], dataType: .float32)
        let mp = mk.dataPointer.assumingMemoryBound(to: Float.self)
        for q in 0..<L { for kk in 0..<L { mp[q * L + kk] = kk < n ? 0 : maskNeg } }
        let pos = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        let pp = pos.dataPointer.assumingMemoryBound(to: Int32.self); for i in 0..<L { pp[i] = Int32(i) }
        let tt = Date()
        let out = try trunk.prediction(from: MLDictionaryFeatureProvider(dictionary: ["inputs_embeds": xe, "mask": mk, "position_ids": pos])).featureValue(for: "out")!.multiArrayValue!
        let msTrunk = Date().timeIntervalSince(tt) * 1000
        let hd = out.shape[2].intValue
        let hidden = Self.floats(out)   // [L * hd], logical order
        guard hidden.prefix(n * hd).contains(where: { $0 != 0 }) else { throw DeciderError(message: "Core ML returned an all-zero trunk output (compute units?)") }
        let z = markers.map { m -> Double in var acc: Float = headB; for j in 0..<min(hd, headW.count) { acc += hidden[m * hd + j] * headW[j] }; return Double(acc) }
        let probs: [Double]
        if multi { probs = z.map { 1 / (1 + exp(-$0)) } } else { let mx = z.max()!; let e = z.map { exp($0 - mx) }; let s = e.reduce(0, +); probs = e.map { $0 / s } }
        func stats(_ v: [Float], _ count: Int) -> String {
            let x = v.prefix(count).filter { !$0.isNaN }
            return String(format: "[%.3g, %.3g] nan %d", x.min() ?? 0, x.max() ?? 0, min(count, v.count) - x.count)
        }
        let diag = "audio \(audioEmb.shape) strides \(audioEmb.strides) \(audioEmb.dataType.rawValue) " + stats(audioRowsF, nAudio * dim)
            + " | trunk \(out.shape) strides \(out.strides) \(out.dataType.rawValue) " + stats(hidden, n * hd)
        return FastResult(probabilities: probs, tokens: n, audioTokens: nAudio, audioSeconds: Double(clip.count) / 16000,
                          msAudio: msAudio, msTrunk: msTrunk, msTotal: Date().timeIntervalSince(t0) * 1000, logits: z, diagnostics: diag)
    }
}
