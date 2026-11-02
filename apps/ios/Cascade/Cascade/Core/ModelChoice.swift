import Foundation

/// What the user chose to install and use (Settings › Models, and the first-run "Get Smaug-Decisions" card).
enum ModelChoice: String, CaseIterable, Codable, Sendable, Identifiable {
    case audioOnly = "audio_only"     // the audio model alone: audio only, nothing to download
    case textImage = "text_image"     // Smaug-Decisions for text + images (no-audio variant); audio stays with the audio model, no escalation
    case audioText = "audio_text"     // Smaug-Decisions for audio + text (no-vision variant); no images
    case full                         // the full cascade (default where the device allows Smaug-Decisions)

    var id: String { rawValue }

    enum Modality: String, Codable, Sendable, CaseIterable { case audio, text, image }

    /// The modalities Smaug-Decisions must have for this choice.
    var bigModalities: Set<Modality> {
        switch self {
        case .audioOnly: return []
        case .textImage: return [.text, .image]
        case .audioText: return [.audio, .text]
        case .full: return [.audio, .text, .image]
        }
    }

    var usesBig: Bool { !bigModalities.isEmpty }
    /// Unsure audio escalates to Smaug-Decisions only when Smaug-Decisions has audio.
    var escalatesAudio: Bool { bigModalities.contains(.audio) }
    func answers(_ kind: Router.InputKind) -> Bool {
        switch kind {
        case .audio: return true   // the audio model always answers audio
        case .text: return bigModalities.contains(.text)
        case .image: return bigModalities.contains(.image)
        }
    }

    var title: String {
        switch self {
        case .audioOnly: return "0.7B audio model only"
        case .textImage: return "Smaug-Decisions for text + images"
        case .audioText: return "Smaug-Decisions for audio + text"
        case .full: return "Full cascade"
        }
    }

    var detail: String {
        switch self {
        case .audioOnly: return "Audio only · already on this iPhone"
        case .textImage: return "Text + photos · audio: 0.7B audio model"
        case .audioText: return "Unsure audio + text · no photos"
        case .full: return "Audio, text and photos"
        }
    }

    /// Fast-only devices get the audio model only; the others get every choice.
    static func offered(mode: DevicePolicy.Mode) -> [ModelChoice] { mode == .full ? allCases : [.audioOnly] }
    static func defaultChoice(mode: DevicePolicy.Mode) -> ModelChoice { mode == .full ? .full : .audioOnly }

    /// The stored choice (or `--models <choice>`), limited to what the device allows.
    static func current(mode: DevicePolicy.Mode, arguments: [String] = ProcessInfo.processInfo.arguments, defaults: UserDefaults = .standard) -> ModelChoice {
        let arg = arguments.firstIndex(of: "--models").flatMap { $0 + 1 < arguments.count ? ModelChoice(rawValue: arguments[$0 + 1]) : nil }
        let c = arg ?? defaults.string(forKey: storeKey).flatMap(ModelChoice.init(rawValue:)) ?? defaultChoice(mode: mode)
        return offered(mode: mode).contains(c) ? c : defaultChoice(mode: mode)
    }
    static let storeKey = "models.choice"
}

/// The first-run "Choose your setup" screen: shown once, before the home screen, while no Smaug-Decisions file is on the device.
enum FirstRun {
    static let seenKey = "firstrun.setup_seen"
    /// Automation launches never see it (they drive the home screen); `--first-run` shows it regardless.
    static let automationArguments = ["--autorun", "--dl-script", "--snapshots", "--show-models"]

    static func shouldShow(seen: Bool, bigOnDevice: Bool, downloadPending: Bool, arguments: [String] = ProcessInfo.processInfo.arguments) -> Bool {
        if arguments.contains("--first-run") { return true }
        if arguments.contains(where: automationArguments.contains) { return false }
        return !seen && !bigOnDevice && !downloadPending
    }

    /// The option marked "Recommended" and preselected: the device's default choice.
    static func recommended(mode: DevicePolicy.Mode) -> ModelChoice { ModelChoice.defaultChoice(mode: mode) }
}

/// Where Smaug-Decisions' files come from. A small JSON manifest; each variant entry has exactly the fields of the quant build's
/// manifest.json (variant, file, bytes, sha256, modalities, vision_tokens). Around them: `base_url` (a variant downloads from
/// base_url + file), optional `url_overrides` (file -> URL, e.g. a test file), and `estimated` (variants whose bytes and sha256 are
/// placeholders until the build is published; they are shown as estimates and cannot be downloaded).
/// Lookup order: `--manifest <path>` (absolute, or relative to Documents), Documents/models_manifest.json (copy one in with
/// Finder or devicectl to point the app at another build or server), then the bundled Resources/models_manifest.json.
struct ModelManifest: Codable, Sendable, Equatable {
    struct Variant: Codable, Sendable, Equatable, Hashable {
        let variant: String
        let file: String
        let bytes: Int64
        let sha256: String
        let modalities: [String]
        /// Image soft tokens of the vision encoder; null for variants without vision.
        let vision_tokens: Int?
        /// Where to download it (the published manifest has one per file); else base_url + file.
        var url: String? = nil

        var modalitySet: Set<ModelChoice.Modality> { Set(modalities.compactMap(ModelChoice.Modality.init(rawValue:))) }
        /// The one-time weight cache LiteRT-LM writes for this file, estimated from its size. Measured on the Smaug-Decisions-LiteRT files:
        /// a 2.04 GB decoder cache, about 0.69 x the decoder's share of the file (the file minus ~0.30 GB of audio encoder and
        /// ~0.17 GB of vision encoder), plus 0.30 GB (audio) and 0.16 GB (vision) of encoder caches.
        var cacheBytes: Int64 {
            let a = modalitySet.contains(.audio), v = modalitySet.contains(.image)
            let decoderFile = bytes - (a ? 300_000_000 : 0) - (v ? 170_000_000 : 0)
            return Int64(Double(decoderFile) * 0.69) + (a ? 300_000_000 : 0) + (v ? 160_000_000 : 0)
        }
        /// The part of the cache a first text/audio load writes (the vision encoder's comes with the first photo).
        var cacheBytesWithoutVision: Int64 { cacheBytes - (modalitySet.contains(.image) ? 160_000_000 : 0) }
    }

    let version: Int
    let base_url: String
    var url_overrides: [String: String]? = nil
    var estimated: [String]? = nil
    let variants: [Variant]
    /// Where this manifest was read from ("bundled", "Documents", or the --manifest path). Not part of the JSON.
    var source = "bundled"

    enum CodingKeys: String, CodingKey { case version, base_url, url_overrides, estimated, variants }

    func isEstimated(_ v: Variant) -> Bool { estimated?.contains(v.variant) ?? false }
    func url(_ v: Variant) -> URL? { URL(string: url_overrides?[v.file] ?? v.url ?? base_url + v.file) }

    /// The variant for a choice: the smallest one with every modality the choice needs, at the device's image size when the
    /// choice includes images (vis560 on 8 GB+, vis280 on 6 GB phones).
    func variant(for choice: ModelChoice, visionTokens: Int?) -> Variant? {
        let need = choice.bigModalities
        guard !need.isEmpty else { return nil }
        let fits = variants.filter { need.isSubset(of: $0.modalitySet) && (!need.contains(.image) || visionTokens == nil || $0.vision_tokens == visionTokens) }
        return fits.min { ($0.bytes, $0.variant) < ($1.bytes, $1.variant) }
    }

    static var documentsOverride: URL { URL.documentsDirectory.appendingPathComponent("models_manifest.json") }

    static func decode(_ data: Data, source: String) throws -> ModelManifest {
        var m = try JSONDecoder().decode(ModelManifest.self, from: data)
        m.source = source
        return m
    }

    static func load(arguments: [String] = ProcessInfo.processInfo.arguments, bundle: Bundle = .main) -> ModelManifest? {
        var candidates: [(URL, String)] = []
        if let i = arguments.firstIndex(of: "--manifest"), i + 1 < arguments.count {
            let p = arguments[i + 1]
            candidates.append((p.hasPrefix("/") ? URL(fileURLWithPath: p) : URL.documentsDirectory.appendingPathComponent(p), p))
        }
        candidates.append((documentsOverride, "Documents"))
        if let b = bundle.url(forResource: "models_manifest", withExtension: "json") { candidates.append((b, "bundled")) }
        for (url, source) in candidates {
            if let data = try? Data(contentsOf: url), let m = try? decode(data, source: source) { return m }
        }
        return nil
    }
}

/// What switching to a choice does to the files in Documents.
struct ModelPlan: Equatable, Sendable {
    let choice: ModelChoice
    let target: ModelManifest.Variant?
    let download: ModelManifest.Variant?          // the target when it is not on the device
    let delete: [ModelManifest.Variant]           // other Smaug-Decisions variants on the device (deleted only after confirmation)

    static func make(choice: ModelChoice, manifest: ModelManifest?, visionTokens: Int?, installed: (String) -> Bool) -> ModelPlan {
        let target = manifest?.variant(for: choice, visionTokens: visionTokens)
        let others = (manifest?.variants ?? []).filter { $0.file != target?.file && installed($0.file) }
        var seen = Set<String>()
        return ModelPlan(choice: choice, target: target, download: target.flatMap { installed($0.file) ? nil : $0 },
                         delete: others.filter { seen.insert($0.file).inserted })
    }

    var freedBytes: Int64 { delete.reduce(0) { $0 + $1.bytes } }
}
