import Accelerate
import Foundation

/// Smaug-Audio-Decisions-0.7B's token-embedding table (scaled, [vocab, 512]), memory-mapped from the bundle. The app ships it as int8 with one
/// fp32 scale per row (tools/quantize_embed.py: 128 MB + 1 MB instead of 256 MB fp16); a row is dequantised on gather with vDSP.
/// The fp16 table is still readable, for the parity check (`--fast-parity`) and older bundles.
struct EmbeddingTable: @unchecked Sendable {
    enum Format: String, Sendable { case int8, fp16 }
    static let dim = 512
    let format: Format
    let vocab: Int
    private let data: Data       // int8 [vocab * dim] or fp16 [vocab * dim]
    private let scales: Data     // float32 [vocab] (int8 only)

    init(int8 q: Data, scales s: Data) throws {
        guard q.count % Self.dim == 0, s.count == q.count / Self.dim * 4 else {
            throw DeciderError(message: "int8 table \(q.count) B and scales \(s.count) B do not match")
        }
        format = .int8; data = q; scales = s; vocab = q.count / Self.dim
    }

    init(fp16 t: Data) throws {
        guard t.count % (Self.dim * 2) == 0 else { throw DeciderError(message: "fp16 table has \(t.count) B") }
        format = .fp16; data = t; scales = Data(); vocab = t.count / (Self.dim * 2)
    }

    /// The bundled table: int8 + scales when present (the shipped app), else fp16.
    static func bundled(_ bundle: Bundle = .main) throws -> EmbeddingTable {
        if let q = bundle.url(forResource: "embed_tokens_int8", withExtension: "bin"),
           let s = bundle.url(forResource: "embed_tokens_int8_scales", withExtension: "bin") {
            return try EmbeddingTable(int8: Data(contentsOf: q, options: .alwaysMapped), scales: Data(contentsOf: s, options: .alwaysMapped))
        }
        guard let f = bundle.url(forResource: "embed_tokens_fp16", withExtension: "bin") else {
            throw DeciderError(message: "missing resource embed_tokens_int8.bin (or embed_tokens_fp16.bin)")
        }
        return try fp16(at: f)
    }

    static func fp16(at url: URL) throws -> EmbeddingTable { try EmbeddingTable(fp16: Data(contentsOf: url, options: .alwaysMapped)) }

    /// Writes row `id` as float32 into dst[0..<512]; out-of-range ids leave dst untouched.
    func row(_ id: Int, into dst: UnsafeMutablePointer<Float>) {
        guard id >= 0, id < vocab else { return }
        let n = vDSP_Length(Self.dim)
        switch format {
        case .int8:
            var scale: Float = scales.withUnsafeBytes { $0.load(fromByteOffset: id * 4, as: Float.self) }
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: id * Self.dim).assumingMemoryBound(to: Int8.self)
                vDSP_vflt8(src, 1, dst, 1, n)
            }
            vDSP_vsmul(dst, 1, &scale, dst, 1, n)
        case .fp16:
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: id * Self.dim * 2).assumingMemoryBound(to: Float16.self)
                for j in 0..<Self.dim { dst[j] = Float(src[j]) }
            }
        }
    }
}
