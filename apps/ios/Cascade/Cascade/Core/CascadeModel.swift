import Combine
import Foundation
import UIKit

/// The app's state and the cascade itself. The fast tier loads at launch; Smaug-Decisions loads on the first escalation (its 3.45 GB file is
/// memory-mapped by LiteRT-LM) and is released under memory pressure. Each tier runs on its own serial queue, one question at a time.
@MainActor
final class CascadeModel: ObservableObject {

    enum BigState: Equatable {
        case notLoaded, loading, loaded(ms: Double, build: String, vision: Bool), released(reason: String), unavailable(reason: String)
        /// The one-time first load of a model file (LiteRT-LM writes its weight cache): started at, expected seconds.
        case preparing(start: Date, expected: Double)
        var label: String {
            switch self {
            case .notLoaded: return "loads on first escalation"
            case .loading: return "loading…"
            case .preparing: return "preparing (one-time)…"
            case .loaded(let ms, let b, let v): return String(format: "%@ loaded (%.1f s), vision %@", b, ms / 1000, v ? "attached" : "not attached")
            case .released(let r): return "released (\(r))"
            case .unavailable(let r): return r
            }
        }
        var isLoaded: Bool { if case .loaded = self { return true }; return false }
    }

    @Published var tau = Router.defaultTau
    @Published private(set) var policy: DevicePolicy?
    @Published private(set) var memory = MemorySnapshot.now()
    @Published private(set) var fastState = "loading…"
    @Published private(set) var fastReady = false
    @Published private(set) var bigState = BigState.notLoaded
    @Published private(set) var outcomes: [Outcome] = []
    @Published private(set) var busy = false
    @Published var status = ""
    @Published private(set) var file: CascadeFile?
    @Published private(set) var tokenizerCheck: String?
    /// Where the list should scroll ("top", "latest"); set after each decision and by autorun.
    @Published private(set) var scroll = ScrollRequest(target: "", n: 0)
    struct ScrollRequest: Equatable { let target: String; let n: Int }
    func scroll(to t: String) { scroll = ScrollRequest(target: t, n: scroll.n + 1) }

    private var fast: FastEngine?
    private var big: BigEngine?
    private var releaseWhenIdle: String?
    private let fastQueue = DispatchQueue(label: "ai.abacus.cascade.fast", qos: .userInitiated)
    private let bigQueue = DispatchQueue(label: "ai.abacus.cascade.big", qos: .userInitiated)
    private var timer: Timer?
    private var pressure: DispatchSourceMemoryPressure?
    private let ramOverrideGB: Double?
    private let forcedMode: DevicePolicy.Mode?
    private let forcedBuild: DevicePolicy.Build?
    /// Weight-cache size before the first Smaug-Decisions load and after it (first run vs warm), MB.
    private(set) var cacheBeforeLoadMB: Double?, cacheAfterLoadMB: Double?
    private var fastLoaded: [CheckedContinuation<Void, Never>] = []

    /// Settings › Models: what is installed and used, and where the files come from.
    @Published private(set) var choice = ModelChoice.full
    @Published private(set) var manifest = ModelManifest.load()
    let downloader = ModelDownloader.shared
    private var subscriptions: Set<AnyCancellable> = []
    /// The --verify-files result.
    private(set) var verifyResult: [[String: Any]]?
    /// The --fast-parity result (int8 vs fp16 embedding table), for the autorun summary.
    private(set) var fastParityResult: [String: Any]?

    init() {
        let args = ProcessInfo.processInfo.arguments
        func arg(_ name: String) -> String? { args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        ramOverrideGB = arg("--ram-gb").flatMap(Double.init)
        forcedMode = arg("--mode").flatMap { $0 == "fast-only" ? .fastOnly : $0 == "full" ? .full : nil }
        forcedBuild = arg("--build").flatMap(DevicePolicy.Build.init(rawValue:))
        if args.contains("--clear-cache"), let dir = try? BigEngine.cacheDir() { try? FileManager.default.removeItem(atPath: dir); ModelPrep.clearAll() }   // first-run measurement
        // --drop-cache <substring>: delete only the cache files whose name contains it (e.g. the CPU decoder cache after switching
        // to the GPU backend; the phone has no room for both)
        if let sub = arg("--drop-cache"), let dir = try? BigEngine.cacheDir() {
            for f in (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [] where f.contains(sub) { try? FileManager.default.removeItem(atPath: dir + "/" + f) }
        }
        if let t = arg("--tau").flatMap(Double.init) { tau = t }
        // --delete-documents a,b: remove those files from Documents (e.g. a replaced model build; devicectl cannot delete)
        for n in (arg("--delete-documents") ?? "").split(separator: ",") where !n.contains("/") {
            try? FileManager.default.removeItem(at: URL.documentsDirectory.appendingPathComponent(String(n)))
        }
        // the weight cache of a model file that is no longer here is dead weight (2-3 GB)
        let models = Set(((try? FileManager.default.contentsOfDirectory(atPath: URL.documentsDirectory.path)) ?? []).filter { $0.hasSuffix(".litertlm") })
        BigEngine.dropStaleCaches(keeping: models)
        do { file = try CascadeFile.load() } catch { fastState = "cascade.json missing: \(error.localizedDescription)" }
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.memory = MemorySnapshot.now() }
        }
        let src = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
        src.setEventHandler { [weak self] in
            Task { @MainActor in self?.memoryPressure(src.data.contains(.critical) ? "critical memory pressure" : "memory pressure warning", release: src.data.contains(.critical)) }
        }
        src.resume(); pressure = src
        NotificationCenter.default.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.memoryPressure("memory warning", release: true) }
        }
        downloader.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &subscriptions)
        downloader.$phase.receive(on: RunLoop.main).sink { [weak self] _ in self?.refreshBigState() }.store(in: &subscriptions)
        downloader.onInstalled = { [weak self] file in
            guard let self else { return }
            self.status = "Smaug-Decisions downloaded and verified (\(file))."
            BigEngine.dropStaleCaches(keeping: self.installedFiles)
            self.refreshBigState()
        }
        migrationTask = Task { await self.migrateRenamedFiles() }
        loadFast()
        runDownloadScript()
    }

    // MARK: loading

    private func loadFast() {
        guard let file else { return }
        let marker = file.marker_id, audioId = file.audio_token_id
        let t0 = Date()
        let progress: @Sendable (String) -> Void = { [weak self] msg in Task { @MainActor in self?.fastState = msg } }
        fastQueue.async { [weak self] in
            do {
                let e = try FastEngine(markerId: marker, audioTokenId: audioId, progress: progress)
                Task { @MainActor in
                    guard let self else { return }
                    self.fast = e; self.fastReady = true
                    self.fastState = String(format: "loaded (%.1f s)", Date().timeIntervalSince(t0))
                    self.memory = MemorySnapshot.now()
                    self.policy = self.evaluatePolicy()
                    self.choice = ModelChoice.current(mode: self.policy?.mode ?? .fastOnly)
                    self.refreshBigState()
                    self.fastLoaded.forEach { $0.resume() }; self.fastLoaded.removeAll()
                }
            } catch { Task { @MainActor in self?.fastState = "load failed: \(error.localizedDescription)" } }
        }
    }

    func waitForFast() async {
        if fastReady { return }
        await withCheckedContinuation { fastLoaded.append($0) }
    }

    private func evaluatePolicy() -> DevicePolicy {
        DevicePolicy.evaluate(memory: MemorySnapshot.now(), ramOverrideGB: ramOverrideGB, forcedMode: forcedMode, forcedBuild: forcedBuild)
    }

    // MARK: models (choice, files, download)

    static func installed(_ file: String) -> Bool { FileManager.default.fileExists(atPath: URL.documentsDirectory.appendingPathComponent(file).path) }
    var installedFiles: Set<String> { Set((manifest?.variants ?? []).map(\.file).filter(Self.installed)) }
    /// Image soft tokens for this device's RAM class (560 on 8 GB+, 280 on 6 GB phones).
    var visionTokens: Int? { policy?.build?.visualTokens }
    var offeredChoices: [ModelChoice] { ModelChoice.offered(mode: policy?.mode ?? .fastOnly) }

    func plan(for c: ModelChoice) -> ModelPlan {
        ModelPlan.make(choice: c, manifest: manifest, visionTokens: visionTokens, installed: Self.installed)
    }

    /// The Smaug-Decisions file the current choice uses: its own variant when installed, else an installed variant that covers it (for
    /// example the full build while a smaller one downloads).
    var activeVariant: ModelManifest.Variant? {
        guard choice.usesBig, let manifest else { return nil }
        if let t = plan(for: choice).target, Self.installed(t.file) { return t }
        let need = choice.bigModalities
        return manifest.variants.filter {
            need.isSubset(of: $0.modalitySet) && Self.installed($0.file) && (!need.contains(.image) || visionTokens == nil || $0.vision_tokens == visionTokens)
        }.min { $0.bytes < $1.bytes }
    }

    /// The variant the "Get Smaug-Decisions" card offers: the device allows Smaug-Decisions, the choice uses it, and no usable file is here.
    var pendingDownload: ModelManifest.Variant? {
        guard !migrating, policy?.mode == .full, choice.usesBig, activeVariant == nil else { return nil }
        return plan(for: choice).target
    }

    /// The same, short enough for one line.
    func downloadBlockerShort(_ v: ModelManifest.Variant) -> String? {
        if manifest?.isEstimated(v) ?? true { return "Not published yet" }
        let already = downloader.item?.file == v.file ? downloader.received : 0
        return ModelStorage.shortShortfall(need: ModelStorage.downloadNeed(v, alreadyDownloaded: already), free: ModelStorage.freeBytes())
    }

    /// A build renamed in the manifest (an older file name -> smaug-decisions-full-vis280): an unlisted .litertlm in Documents with a
    /// listed variant's size and sha256 is renamed to it, together with its weight cache (the move keeps the mtime that LiteRT-LM
    /// puts in the cache names), its remembered backend and its prepared flag, instead of downloading it again.
    @Published private(set) var migrating = false
    private var migrationTask: Task<[String], Never>?
    private(set) var migrated: [String] = []

    func migrateRenamedFiles() async -> [String] {
        guard let manifest else { return [] }
        let fm = FileManager.default
        let listed = Set(manifest.variants.map(\.file))
        let orphans = ((try? fm.contentsOfDirectory(atPath: URL.documentsDirectory.path)) ?? []).filter { $0.hasSuffix(".litertlm") && !listed.contains($0) }
        guard !orphans.isEmpty else { return [] }
        migrating = true
        defer { migrating = false; status = ""; refreshBigState() }
        var done: [String] = []
        for old in orphans {
            let u = URL.documentsDirectory.appendingPathComponent(old)
            let size = ((try? fm.attributesOfItem(atPath: u.path)[.size]) as? NSNumber)?.int64Value ?? -1
            let candidates = manifest.variants.filter { $0.bytes == size && !Self.installed($0.file) }
            guard !candidates.isEmpty else { continue }
            status = "Checking \(old)…"
            guard let r = try? await Task.detached(operation: { try ModelDownloader.sha256(of: u) { _ in } }).value,
                  let v = candidates.first(where: { $0.sha256.lowercased() == r.0 }) else { continue }
            do { try fm.moveItem(at: u, to: URL.documentsDirectory.appendingPathComponent(v.file)) } catch { continue }
            if let dir = try? BigEngine.cacheDir() {
                for n in (try? fm.contentsOfDirectory(atPath: dir)) ?? [] where n.hasPrefix(old + "_") {
                    try? fm.moveItem(atPath: dir + "/" + n, toPath: dir + "/" + v.file + n.dropFirst(old.count))
                }
            }
            let d = UserDefaults.standard
            for b in ["cpu", "gpu"] where d.bool(forKey: ModelPrep.key(file: old, bytes: size, backend: b)) {
                d.set(true, forKey: ModelPrep.key(file: v.file, bytes: v.bytes, backend: b))
            }
            if let b = d.string(forKey: "smaug.backend.\(old).\(size)") { d.set(b, forKey: Self.backendKey(v)) }
            done.append("\(old) -> \(v.file)")
        }
        migrated = done
        return done
    }

    /// Why the pending download cannot start now (not published yet, no room), or nil.
    func downloadBlocker(_ v: ModelManifest.Variant) -> String? {
        if manifest?.isEstimated(v) ?? true { return "This Smaug-Decisions variant cannot be downloaded yet (its size is an estimate)." }
        let already = downloader.item?.file == v.file ? downloader.received : 0
        return ModelStorage.shortfall(need: ModelStorage.downloadNeed(v, alreadyDownloaded: already), free: ModelStorage.freeBytes(), what: "Smaug-Decisions")
    }

    func startDownload(wifiOnly: Bool = false) {
        guard let v = pendingDownload, let url = manifest?.url(v) else { return }
        if let why = downloadBlocker(v) { status = why; return }
        downloader.start(.init(file: v.file, url: url.absoluteString, sha256: v.sha256, bytes: v.bytes, wifiOnly: wifiOnly))
    }

    /// Switches the models. Files of other variants are deleted only when `deleteOthers` (the user confirmed); a running download
    /// of a file the new choice does not need is cancelled.
    func apply(_ c: ModelChoice, deleteOthers: Bool) {
        guard offeredChoices.contains(c) else { return }
        UserDefaults.standard.set(c.rawValue, forKey: ModelChoice.storeKey)
        let p = plan(for: c)
        choice = c
        if let big, big.modelPath != activeVariant.map({ URL.documentsDirectory.appendingPathComponent($0.file).path }) { releaseBig("models changed") }
        if let item = downloader.item, item.file != p.download?.file, downloader.phase.active || downloader.resumable(item.file) { downloader.cancel() }
        if deleteOthers {
            for v in p.delete { try? FileManager.default.removeItem(at: URL.documentsDirectory.appendingPathComponent(v.file)) }
            let freed = BigEngine.dropStaleCaches(keeping: installedFiles)
            status = String(format: "Deleted %d file(s), freed %@.", p.delete.count, ModelStorage.gb(p.freedBytes + freed))
        }
        refreshBigState()
    }

    private func refreshBigState() {
        guard let policy else { return }
        switch bigState { case .loading, .preparing, .loaded: return; default: break }
        if policy.mode == .fastOnly { bigState = .unavailable(reason: "not loaded: Fast-only mode") }
        else if !choice.usesBig { bigState = .unavailable(reason: "not installed (0.7B audio model only)") }
        else if activeVariant == nil {
            bigState = .unavailable(reason: downloader.phase.active ? "downloading (\(downloader.percent)%)" : "not on this iPhone yet")
        } else if case .unavailable = bigState { bigState = .notLoaded }
    }

    /// Why Smaug-Decisions cannot answer right now because of its files (downloading, missing), or nil.
    private var missingBigNote: String {
        switch downloader.phase {
        case .downloading: return "Smaug-Decisions is downloading (\(downloader.percent)%)"
        case .verifying: return "Smaug-Decisions is downloading (verifying the file)"
        default: return "Smaug-Decisions not downloaded yet"
        }
    }

    /// The installed Smaug-Decisions file has not been through its one-time first load yet (for the "Prepare now" button).
    var bigNeedsPrepare: Bool {
        guard let v = activeVariant else { return false }
        return !ModelPrep.isPrepared(file: v.file, bytes: v.bytes, backend: Self.backend(for: v))
    }

    /// "Prepare now" on the card after a download: the one-time first load, before the first question needs it.
    func prepareNow() {
        guard !busy else { return }
        busy = true
        Task {
            let (e, _, why) = await bigEngine(vision: false)
            busy = false
            if e == nil, let why { status = why } else { status = "Smaug-Decisions is ready." }
            if let r = releaseWhenIdle { releaseWhenIdle = nil; releaseBig(r) }
        }
    }

    /// `--threads n` / `--backend gpu` (launch arguments) for experiments.
    private static let argThreads = ProcessInfo.processInfo.arguments.firstIndex(of: "--threads").flatMap { i in
        i + 1 < ProcessInfo.processInfo.arguments.count ? Int(ProcessInfo.processInfo.arguments[i + 1]) : nil }
    static let backend = ProcessInfo.processInfo.arguments.firstIndex(of: "--backend").flatMap { i in
        i + 1 < ProcessInfo.processInfo.arguments.count ? ProcessInfo.processInfo.arguments[i + 1] : nil } ?? defaultBackend
    /// The decoder backend. On the iPhone 14 Pro Max the GPU backend was faster than the CPU (text 0.6-1.7 s vs 2.1-7.4 s
    /// prefill, audio 2.8 s vs 4.1-12 s) with the same answers, so phones use it; the encoders stay on the CPU. The Simulator
    /// uses the CPU (its GPU path is untested). Each backend writes its own ~2.2 GB weight cache.
    #if targetEnvironment(simulator)
    static let defaultBackend = "cpu"
    #else
    static let defaultBackend = "gpu"
    #endif

    private static var threads: Int {
        if let argThreads { return argThreads }
        let cores = ProcessInfo.processInfo.activeProcessorCount
        #if targetEnvironment(simulator)
        return min(8, cores)
        #else
        return min(4, cores)
        #endif
    }

    /// Smaug-Decisions, loading it on first use when the device mode and the memory left allow, with the vision encoder attached only for an
    /// image (the engine is recreated with it; LiteRT-LM fixes the encoders at engine creation). Returns nil (and why) when it
    /// cannot be used.
    private func bigEngine(vision: Bool) async -> (BigEngine?, loadMs: Double?, note: String?) {
        guard let policy, policy.mode == .full, let build = policy.build else { return (nil, nil, "Fast-only mode: Smaug-Decisions does not fit this device") }
        guard let variant = activeVariant else { return (nil, nil, missingBigNote) }
        let vision = vision && variant.modalitySet.contains(.image)
        if let big, big.hasVision || !vision { return (big, nil, nil) }
        let url = URL.documentsDirectory.appendingPathComponent(variant.file)
        // Re-check the memory left now (other apps, or a phone where the estimate was optimistic) before mapping the file / attaching
        // the vision encoder.
        let now = evaluatePolicy()
        let need = (big == nil ? build.loadNeedMB : 0) + (vision ? build.visionNeedMB : 0)
        if !policy.forced, now.availableMB < need {
            return (nil, nil, String(format: "only %.0f MB available, Smaug-Decisions%@ needs %.0f MB", now.availableMB, vision ? " with an image" : "", need))
        }
        // Storage: the first load (and the first photo) write the weight cache. Refuse with the numbers instead of failing midway.
        // A file only loads once it is complete (a copy or download that stopped midway has the wrong size).
        let size = ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? -1
        if size != variant.bytes {
            lastLoadError = "\(variant.file): \(size) bytes on disk, the manifest says \(variant.bytes)"
            return (nil, nil, "Smaug-Decisions file is incomplete")
        }
        var backend = Self.backend(for: variant)
        let prepared = ModelPrep.isPrepared(file: variant.file, bytes: variant.bytes, backend: backend)
        let existing = ModelPrep.existingCache(file: variant.file, flagged: prepared)
        let toWrite = (vision ? variant.cacheBytes : variant.cacheBytesWithoutVision) - existing
        if !prepared || toWrite > 100_000_000 {
            let need = ModelStorage.firstLoadNeed(variant, existingCache: existing, vision: vision)
            if let msg = ModelStorage.shortfall(need: need, free: ModelStorage.freeBytes(), what: prepared ? "Attaching Smaug-Decisions' vision encoder" : "Preparing Smaug-Decisions") {
                return (nil, nil, msg)
            }
        }
        let attaching = big != nil
        if let old = big {   // drop the text/audio engine first so the two are never resident together
            big = nil
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in bigQueue.async { _ = old; c.resume() } }
        }
        if prepared {
            bigState = .loading
            status = attaching ? "Attaching the vision encoder…" : "Loading Smaug-Decisions (\(variant.variant))…"
        } else {
            let expected = ModelPrep.expectedSeconds(fileBytes: variant.bytes, backend: backend)
            bigState = .preparing(start: Date(), expected: expected)
            status = String(format: "Preparing Smaug-Decisions (one-time, about %.0f s)…", expected)
        }
        if cacheBeforeLoadMB == nil { cacheBeforeLoadMB = BigEngine.cacheMB() }
        let t0 = DispatchTime.now(), path = url.path, threads = Self.threads
        let vt = variant.vision_tokens ?? build.visualTokens
        do {
            let e: BigEngine
            do {
                let be = backend
                e = try await onQueue(bigQueue) { try BigEngine(modelPath: path, threads: threads, vision: vision, visualTokens: vt, backend: be) }
            } catch where backend == "gpu" && !Self.backendForced {
                // The GPU delegate cannot take every build (e.g. a build whose letter head uses gather ops: "failed to modify
                // graph"). Retry on the CPU (XNNPack) and remember it for this file.
                gpuFallbackError = error.localizedDescription
                backend = "cpu"
                UserDefaults.standard.set("cpu", forKey: Self.backendKey(variant))
                status = "Smaug-Decisions: the GPU backend cannot run this build, loading on the CPU…"
                e = try await onQueue(bigQueue) { try BigEngine(modelPath: path, threads: threads, vision: vision, visualTokens: vt, backend: "cpu") }
            }
            lastBackend = backend
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            if cacheAfterLoadMB == nil { cacheAfterLoadMB = BigEngine.cacheMB() }
            if !prepared {
                ModelPrep.markPrepared(file: variant.file, bytes: variant.bytes, backend: backend, seconds: ms / 1000)
                lastPrepareSeconds = ms / 1000
            }
            big = e; bigState = .loaded(ms: ms, build: variant.variant, vision: vision)
            return (e, ms, nil)
        } catch {
            lastLoadError = error.localizedDescription + (gpuFallbackError.map { " (GPU first: \($0))" } ?? "")
            bigState = .unavailable(reason: "could not load")
            return (nil, nil, "Smaug-Decisions could not load")
        }
    }

    /// Raw runtime text of the last failed load (shown in Details), the GPU error that triggered a CPU fallback, the backend used.
    private(set) var lastLoadError: String?, gpuFallbackError: String?, lastBackend: String?

    static let backendForced = ProcessInfo.processInfo.arguments.contains("--backend")
    static func backendKey(_ v: ModelManifest.Variant) -> String { "smaug.backend.\(v.file).\(v.bytes)" }
    /// The backend for a model file: `--backend` if given, else what worked for this file before, else the platform default.
    static func backend(for v: ModelManifest.Variant) -> String {
        if backendForced { return backend }
        return UserDefaults.standard.string(forKey: backendKey(v)) ?? backend
    }

    /// Seconds the one-time prepare took in this process (for the autorun summary).
    private(set) var lastPrepareSeconds: Double?

    /// Drops Smaug-Decisions (the engine and its mapped weights); the next escalation loads it again.
    func releaseBig(_ reason: String) {
        guard big != nil else { return }
        if busy { releaseWhenIdle = reason; return }
        let e = big; big = nil
        bigQueue.async { _ = e }   // deinit (engine delete) on the engine's own queue, after any queued work
        bigState = .released(reason: reason)
        status = "Smaug-Decisions released (\(reason))"
    }

    /// Every pressure event, for the autorun report.
    private(set) var pressureEvents: [[String: Any]] = []

    /// Releases Smaug-Decisions on critical system memory pressure or on UIKit's memory warning to this app. A system-wide pressure *warning*
    /// is only logged: on the iPhone 14 Pro Max it fired after ordinary decisions at a 1.2 GB footprint (the limit is 4 GB), and
    /// releasing there made every escalation reload Smaug-Decisions.
    private func memoryPressure(_ reason: String, release: Bool) {
        let m = MemorySnapshot.now()
        pressureEvents.append(["reason": reason, "released": release && big != nil, "footprint_mb": m.footprintMB,
                               "available_mb": m.availableMB ?? -1, "big_loaded": big != nil, "t": Date().timeIntervalSince1970])
        if release { releaseBig(reason) }
    }

    // MARK: the cascade

    @discardableResult
    func run(_ q: Question) async -> Outcome? {
        guard !busy else { return nil }
        await waitForFast()
        _ = await migrationTask?.value
        guard let fast else { status = "fast tier not loaded"; return nil }
        busy = true
        defer {
            busy = false
            if let r = releaseWhenIdle { releaseWhenIdle = nil; releaseBig(r) }
        }
        let tau = self.tau
        var fastResult: FastResult?, conf: Double?, note: String?
        if Router.startsOnFast(q.kind) {
            guard let fp = q.fast, let samples = q.samples else { status = "no fast-tier prompt for this question"; return nil }
            status = "Fast tier…"
            do {
                let k = q.options.count, multi = q.multi
                let r = try await onQueue(fastQueue) { try fast.run(fp, options: k, multi: multi, samples: samples) }
                fastResult = r; conf = Router.confidence(r.probabilities, multi: q.multi)
            } catch { status = "fast tier failed: \(error.localizedDescription)"; return nil }
        }
        let choice = self.choice
        var route = Router.route(kind: q.kind, fastConfidence: conf, tau: tau, bigAvailable: true, choice: choice)
        if route == .unsure { note = "Smaug-Decisions not installed for audio" }
        if route == .notInstalled { note = choice.usesBig ? "\(q.kind == .image ? "Images" : "Text") not installed" : "Smaug-Decisions not installed" }
        var bigResult: BigResult?, loadMs: Double?
        if route == .escalated || route == .routed {
            let (engine, ms, why) = await bigEngine(vision: q.kind == .image)
            loadMs = ms
            if let engine {
                status = route == .escalated ? "Escalated to Smaug-Decisions…" : "Smaug-Decisions…"
                do {
                    let media: MediaFile?
                    switch q.kind {
                    case .text: media = nil
                    case .image: media = q.imageURL.map { MediaFile(kind: .image, path: $0.path) }
                    case .audio: media = MediaFile(kind: .audio, path: try MediaPrep.wav(q.samples ?? [], name: "cascade_big_\(q.id).wav").path)
                    }
                    let k = q.options.count, multi = q.multi, image = q.kind == .image
                    let audioSamples = q.kind == .audio ? min(q.samples?.count ?? 0, Prompt.maxAudioSeconds * Prompt.sampleRate) : nil
                    let question = q
                    let (r, budget) = try await onQueue(bigQueue) { () -> (BigResult, Int) in
                        // smaug_decide's prompt at MAX_TOKENS 2000 (the 2048-token cache), shrunk further if the counted tokens overflow
                        let fit = try Prompt.fitted(limit: Prompt.deviceMaxTokens,
                                                    render: { question.bigPrompt(maxTokens: $0, imageTokens: engine.visualTokenBudget) },
                                                    count: { try engine.promptTokens($0, audioSamples: audioSamples, image: image) })
                        return (try engine.decide(prompt: fit.prompt, options: k, multi: multi, media: media), fit.maxTokens)
                    }
                    bigResult = r
                    if budget < Prompt.deviceMaxTokens { note = "prompt trimmed to fit the 2048-token cache (budget \(budget))" }
                } catch { note = "Smaug-Decisions could not answer"; lastLoadError = error.localizedDescription }
            } else { note = why }
            if bigResult == nil { route = Router.route(kind: q.kind, fastConfidence: conf, tau: tau, bigAvailable: false, choice: choice) }
        }
        let mem = MemorySnapshot.now(); memory = mem
        let failure = bigResult == nil && (route == .needsBig || route == .wouldEscalate || note == "Smaug-Decisions could not answer") ? lastLoadError : nil
        var o = Outcome(questionId: q.id, title: q.title, kind: q.kind, question: q.question, options: q.options, multi: q.multi,
                        route: route, tau: tau, fast: fastResult, fastConfidence: conf, big: bigResult, bigLoadMs: loadMs,
                        memory: mem, note: note, fastRef: q.fast?.ref_torch, bigRef: q.bigRef,
                        bigBuild: bigResult == nil ? nil : big.map { "vis\($0.visualTokenBudget), vision \($0.hasVision ? "attached" : "not attached")" })
        o.detail = failure
        outcomes.insert(o, at: 0)
        scroll(to: "latest")
        status = ""   // the result card says it
        return o
    }

    static func describe(_ o: Outcome) -> String {
        switch o.route {
        case .fast: return String(format: "fast tier answered (confidence %.2f ≥ τ %.2f)", o.fastConfidence ?? 0, o.tau)
        case .escalated: return String(format: "escalated: confidence %.2f < τ %.2f, Smaug-Decisions answered", o.fastConfidence ?? 0, o.tau)
        case .routed: return "\(o.kind.rawValue) routed straight to Smaug-Decisions"
        case .wouldEscalate: return String(format: "would escalate (confidence %.2f < τ %.2f): %@", o.fastConfidence ?? 0, o.tau, o.note ?? "")
        case .needsBig: return "needs Smaug-Decisions: \(o.note ?? "")"
        case .unsure: return String(format: "unsure (confidence %.2f < τ %.2f); %@", o.fastConfidence ?? 0, o.tau, o.note ?? "")
        case .notInstalled: return o.note ?? "not installed"
        }
    }

    // MARK: inputs

    func question(for c: DemoCase) -> Question? {
        var q = Question(id: c.id, title: c.title, kind: c.kind, question: c.question, options: c.options, descriptions: c.descriptions,
                         multi: c.multi, rules: c.rules)
        q.fast = c.fast; q.bigRef = c.big_ref; q.bigRefTokens = c.big_ref_tokens
        switch c.kind {
        case .text: q.text = c.text
        case .image: q.imageURL = c.media.flatMap(DemoCase.bundled)
        case .audio:
            guard let m = c.media, let url = DemoCase.bundled(m) else { status = "missing \(c.media ?? "?")"; return nil }
            if url.pathExtension == "pcm" {   // raw 16 kHz float32 (the bundled synthetic clips)
                guard let data = try? Data(contentsOf: url) else { return nil }
                q.samples = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            } else {
                do { q.samples = try MediaPrep.samples16kMono(from: url) } catch { status = error.localizedDescription; return nil }
            }
        }
        return q
    }

    func question(recording samples: [Float], scenario s: FastScenario) -> Question {
        var q = Question(id: "rec", title: "Recording: \(s.title)", kind: .audio, question: s.question, options: s.options,
                         descriptions: s.descriptions, multi: s.multi, rules: s.rules)
        q.samples = samples; q.fast = s.fast
        return q
    }

    /// A fast-tier prompt for any question, tokenized on the phone with Smaug-Decisions' LiteRT-LM tokenizer (same BPE as the audio model; see
    /// FastTokenizer). Needs Smaug-Decisions loaded: the tokenizer lives in the .litertlm file.
    func fastPrompt(question: String, options: [String], multi: Bool) async throws -> FastPrompt {
        guard let file, let big else { throw DeciderError(message: "Free-form questions for the audio model use Smaug-Decisions' tokenizer: load Smaug-Decisions first (escalate once).") }
        let marker = file.marker_id, boa = file.audio_bos_id, eoa = file.audio_eos_id
        return try await onQueue(bigQueue) {
            try FastTokenizer.prompt(question: question, options: options, descriptions: Array(repeating: nil, count: options.count), multi: multi,
                                     rules: nil, marker: marker, audioBOS: boa, audioEOS: eoa, tokenize: big.tokenize)
        }
    }

    var canTokenize: Bool { big != nil }

    func question(recording samples: [Float], question: String, options: [String], multi: Bool) async -> Question? {
        do {
            var q = Question(id: "rec", title: "Recording: free-form", kind: .audio, question: question, options: options,
                             descriptions: Array(repeating: nil, count: options.count), multi: multi, rules: nil)
            q.samples = samples
            q.fast = try await fastPrompt(question: question, options: options, multi: multi)
            return q
        } catch { status = error.localizedDescription; return nil }
    }

    func runCase(_ c: DemoCase) { if let q = question(for: c) { Task { await run(q) } } }

    // MARK: autorun (screenshots and measurements)

    /// `--autorun a1,a2,a3,t1,i1`: runs those cases in order, writes Documents/autorun_<id>.json per case and
    /// Documents/autorun_summary.json. `--tokcheck` also checks the LiteRT-LM tokenizer against the pre-tokenised fast-tier prompts
    /// after Smaug-Decisions is loaded; `--release-at-end` releases Smaug-Decisions afterwards and records the footprint.
    func autorunFromLaunchArguments() async {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--autorun"), i + 1 < args.count, let file else { return }
        await waitForFast()
        _ = await migrationTask?.value
        if args.contains("--fast-parity") { fastParityResult = await fastParity() }
        // --verify-files: sha256 of every installed Smaug-Decisions file against the manifest (after a devicectl copy)
        if args.contains("--verify-files") {
            var res: [[String: Any]] = []
            for v in manifest?.variants ?? [] where Self.installed(v.file) && !res.contains(where: { $0["file"] as? String == v.file }) {
                status = "Verifying \(v.file)…"
                let url = URL.documentsDirectory.appendingPathComponent(v.file)
                let t0 = Date()
                let r = try? await Task.detached { try ModelDownloader.sha256(of: url) { _ in } }.value
                res.append(["file": v.file, "sha256": r?.0 ?? "error", "bytes": r?.1 ?? -1, "ok": r?.0 == v.sha256.lowercased() && r?.1 == v.bytes,
                            "seconds": Date().timeIntervalSince(t0)])
            }
            verifyResult = res
        }
        // --autorun-delay s: start the cases later (e.g. while a download runs)
        if let j = args.firstIndex(of: "--autorun-delay"), j + 1 < args.count, let d = Double(args[j + 1]) { try? await Task.sleep(nanoseconds: UInt64(d * 1e9)) }
        var rows: [[String: Any]] = []
        for id in args[i + 1].split(separator: ",").map(String.init) where !id.isEmpty {
            guard let c = file.cases.first(where: { $0.id == id }), let q = question(for: c) else { continue }
            guard let o = await run(q) else { continue }
            var row = Self.json(o)
            row["fast_ref_coreml_gpu"] = c.fast?.ref_coreml_gpu ?? []
            row["fast_ref_coreml_cpu"] = c.fast?.ref_coreml_cpu ?? []
            row["fast_ref_tokens"] = c.fast?.ref_tokens ?? -1
            row["big_ref_tokens"] = c.big_ref_tokens ?? -1
            row["source"] = c.source
            Self.write(row, "autorun_\(id).json")
            rows.append(row)
            try? await Task.sleep(nanoseconds: 1_500_000_000)   // let the UI settle for the recording
        }
        if args.contains("--tokcheck") { await checkTokenizer() }
        // --freeform "a1|Is a woman speaking?|Yes,No": a question that is not pre-tokenised, on a case's clip, via Smaug-Decisions' tokenizer
        if let j = args.firstIndex(of: "--freeform"), j + 1 < args.count {
            let parts = args[j + 1].components(separatedBy: "|")
            if parts.count == 3, let c = file.cases.first(where: { $0.id == parts[0] }), let base = question(for: c), let samples = base.samples {
                let opts = parts[2].components(separatedBy: ",")
                if var q = await question(recording: samples, question: parts[1], options: opts, multi: false) {
                    q = Question(id: "freeform", title: "Free-form: \(c.title)", kind: .audio, question: q.question, options: q.options,
                                 descriptions: q.descriptions, multi: false, rules: nil, samples: samples, fast: q.fast)
                    if let o = await run(q) {
                        var row = Self.json(o); row["clip_case"] = c.id; row["fast_pre_ids"] = q.fast?.pre_ids ?? []; row["fast_post_ids"] = q.fast?.post_ids ?? []
                        Self.write(row, "autorun_freeform.json"); rows.append(row)
                    }
                }
            }
        }
        var summary: [String: Any] = ["cases": rows, "tau": tau, "memory_end": MemorySnapshot.now().dict, "fast_state": fastState,
                                      "big_state": bigState.label, "tokenizer_check": tokenizerCheck ?? "not run"]
        if let p = policy {
            summary["policy"] = ["mode": p.mode.rawValue, "physical_mb": p.physicalMB, "limit_mb": p.limitMB, "available_mb": p.availableMB,
                                 "limit_source": p.limitSource, "forced": p.forced, "build": p.build?.rawValue ?? "none",
                                 "load_need_mb": p.build?.loadNeedMB ?? 0, "vision_need_mb": p.build?.visionNeedMB ?? 0]
        }
        summary["cache_mb_before_first_load"] = cacheBeforeLoadMB ?? -1
        summary["cache_mb_after_first_load"] = cacheAfterLoadMB ?? -1
        summary["cache_mb_end"] = BigEngine.cacheMB()
        summary["peak_footprint_mb"] = MemorySnapshot.now().footprintPeakMB
        summary["disk_free_gb"] = Self.diskFreeGB()
        summary["big_threads"] = Self.threads; summary["big_backend"] = lastBackend ?? Self.backend
        summary["big_backend_default"] = Self.backend; summary["gpu_fallback_error"] = gpuFallbackError ?? ""; summary["last_load_error"] = lastLoadError ?? ""
        summary["pressure_events"] = pressureEvents
        if let r = fastParityResult { summary["fast_parity"] = r }
        if let r = verifyResult { summary["verify_files"] = r }
        summary["fast_table"] = fast?.table.format.rawValue ?? "?"
        if let j = args.firstIndex(of: "--wait-download"), j + 1 < args.count, let n = Double(args[j + 1]) {
            let t0 = Date()
            while Date().timeIntervalSince(t0) < n {
                if case .done = downloader.phase { break }
                if case .failed = downloader.phase { break }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        summary["models"] = modelsReport()
        summary["documents"] = (try? FileManager.default.contentsOfDirectory(atPath: URL.documentsDirectory.path)) ?? []
        if let p = policy, p.mode == .fastOnly {
            summary["note"] = "fast only"
        }
        if args.contains("--release-at-end") {
            releaseBig("autorun end")
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            summary["memory_after_release"] = MemorySnapshot.now().dict
        }
        #if targetEnvironment(simulator)
        summary["platform"] = "iOS Simulator"
        #else
        summary["platform"] = "iOS device"
        #endif
        Self.write(summary, "autorun_summary.json")
        status = ""   // the summary is in Documents; no debug line on the main screen
        if let j = args.firstIndex(of: "--end-scroll"), j + 1 < args.count { scroll(to: args[j + 1]) }
        // --snapshot name: the app renders its own window to Documents/snapshot_<name>.png (device screenshots without extra tools)
        if let j = args.firstIndex(of: "--snapshot"), j + 1 < args.count {
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            Self.snapshot(args[j + 1])
            Self.write(["snapshot": args[j + 1]], "snapshot_done.json")
        }
        if args.contains("--dl-cleanup") { Self.write(["removed": cleanUpTestDownload(), "models": modelsReport()], "dl_cleanup.json") }
        // --watch N: write Documents/state.json every second for N s (Smaug-Decisions state + memory), e.g. to see a memory warning release Smaug-Decisions
        if let j = args.firstIndex(of: "--watch"), j + 1 < args.count, let n = Int(args[j + 1]) {
            for t in 0..<n {
                Self.write(["t": t, "big_state": bigState.label, "memory": MemorySnapshot.now().dict], "state.json")
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    // MARK: automation for the download flow and the audio model parity check

    /// Phase changes of the downloader with time and bytes, for the autorun summary.
    private var dlTimeline: [[String: Any]] = []
    private let launched = Date()

    /// `--dl-script "start@2,pause@6,resume@9,prepare@40"` drives the download (and the prepare) at those seconds after launch;
    /// `--snapshots card@2,dl@7` saves window snapshots then.
    private func runDownloadScript() {
        let args = ProcessInfo.processInfo.arguments
        func steps(_ name: String) -> [(String, Double)] {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else { return [] }
            return args[i + 1].split(separator: ",").compactMap { p in
                let a = p.split(separator: "@"); return a.count == 2 ? (String(a[0]), Double(a[1]) ?? 0) : nil
            }
        }
        downloader.$phase.sink { [weak self] ph in
            guard let self else { return }
            self.dlTimeline.append(["t": Date().timeIntervalSince(self.launched), "phase": "\(ph)", "received": self.downloader.received])
        }.store(in: &subscriptions)
        for (what, t) in steps("--dl-script") {
            Task {
                try? await Task.sleep(nanoseconds: UInt64(t * 1e9))
                await self.waitForFast()
                switch what {
                case "start", "resume": self.startDownload()
                case "pause": self.downloader.pause()
                case "prepare": self.prepareNow()
                default: break
                }
                self.dlTimeline.append(["t": Date().timeIntervalSince(self.launched), "step": what, "received": self.downloader.received])
            }
        }
        for (name, t) in steps("--snapshots") {
            Task {
                try? await Task.sleep(nanoseconds: UInt64(t * 1e9))
                Self.snapshot(name)
                self.dlTimeline.append(["t": Date().timeIntervalSince(self.launched), "snapshot": name, "received": self.downloader.received])
            }
        }
    }

    func modelsReport() -> [String: Any] {
        var r: [String: Any] = ["choice": choice.rawValue, "offered": offeredChoices.map(\.rawValue), "manifest_source": manifest?.source ?? "none",
                                "installed": Array(installedFiles).sorted(), "active_variant": activeVariant?.variant ?? "none",
                                "pending_download": pendingDownload?.variant ?? "none", "download_phase": "\(downloader.phase)",
                                "download_received": downloader.received, "download_timeline": dlTimeline,
                                "disk_free_bytes": ModelStorage.freeBytes() ?? -1, "prepare_seconds": lastPrepareSeconds ?? -1,
                                "migrated": migrated]
        if let v = pendingDownload { r["download_blocker"] = downloadBlocker(v) ?? "none"; r["download_need_bytes"] = ModelStorage.downloadNeed(v) }
        if let i = downloader.item {
            let u = ModelDownloader.finalURL(i.file)
            r["download_item"] = ["file": i.file, "bytes": i.bytes, "sha256": i.sha256, "url": i.url]
            r["download_final_exists"] = FileManager.default.fileExists(atPath: u.path)
            r["download_final_bytes"] = (try? FileManager.default.attributesOfItem(atPath: u.path)[.size] as? Int64) ?? -1
            if let s = downloader.startedAt { r["download_seconds"] = Date().timeIntervalSince(s) }
        }
        return r
    }

    /// `--dl-cleanup`: removes Documents/models_manifest.json and the files only it named (a test download). Never a file the
    /// bundled manifest names.
    private func cleanUpTestDownload() -> [String] {
        guard manifest?.source != "bundled" else { return [] }
        let bundledFiles = Set((Bundle.main.url(forResource: "models_manifest", withExtension: "json").flatMap { try? Data(contentsOf: $0) }
            .flatMap { try? ModelManifest.decode($0, source: "bundled") }?.variants ?? []).map(\.file))
        var removed: [String] = []
        for v in manifest?.variants ?? [] where !bundledFiles.contains(v.file) && Self.installed(v.file) {
            try? FileManager.default.removeItem(at: URL.documentsDirectory.appendingPathComponent(v.file)); removed.append(v.file)
        }
        if (try? FileManager.default.removeItem(at: ModelManifest.documentsOverride)) != nil { removed.append("models_manifest.json") }
        manifest = ModelManifest.load()
        refreshBigState()
        return removed
    }

    /// `--fast-parity`: every fast-tier prompt (the 6 scenarios + the 4 audio cases) on every bundled clip, once with the bundled
    /// int8 embedding table and once with the fp16 table (Documents/embed_tokens_fp16.bin, or the bundle's), same compute units.
    /// `--fast-parity-cleanup` deletes the fp16 copy from Documents afterwards.
    func fastParity() async -> [String: Any] {
        guard let fast, let file else { return ["error": "fast tier not loaded"] }
        let docFP16 = URL.documentsDirectory.appendingPathComponent("embed_tokens_fp16.bin")
        let fp16URL = FileManager.default.fileExists(atPath: docFP16.path) ? docFP16 : Bundle.main.url(forResource: "embed_tokens_fp16", withExtension: "bin")
        guard let fp16URL, let ref = try? EmbeddingTable.fp16(at: fp16URL) else { return ["error": "no fp16 table (copy embed_tokens_fp16.bin to Documents)"] }
        status = "Audio model parity: int8 vs fp16 table…"
        let clips: [(String, [Float])] = file.cases.filter { $0.kind == .audio }.compactMap { c in question(for: c)?.samples.map { (c.id, $0) } }
        let prompts: [(String, FastPrompt, Int, Bool)] = file.scenarios.map { ($0.id, $0.fast, $0.options.count, $0.multi) }
            + file.cases.compactMap { c in c.fast.map { (c.id, $0, c.options.count, c.multi) } }
        var rows: [[String: Any]] = [], dps: [Double] = [], skipped = 0, topSame = 0, routeSame = 0, maxDz = 0.0
        for (clipId, samples) in clips {
            for (pid, fp, k, multi) in prompts {
                do {
                    let (a, b) = try await onQueue(fastQueue) {
                        (try fast.run(fp, options: k, multi: multi, samples: samples), try fast.run(fp, options: k, multi: multi, samples: samples, table: ref))
                    }
                    let dp = zip(a.probabilities, b.probabilities).map { abs($0 - $1) }.max() ?? 0
                    let dz = zip(a.logits, b.logits).map { abs($0 - $1) }.max() ?? 0
                    let argmax = { (p: [Double]) in p.indices.max { p[$0] < p[$1] }! }
                    let same = multi ? zip(a.probabilities, b.probabilities).allSatisfy { ($0 > 0.5) == ($1 > 0.5) } : argmax(a.probabilities) == argmax(b.probabilities)
                    let ca = Router.confidence(a.probabilities, multi: multi), cb = Router.confidence(b.probabilities, multi: multi)
                    dps.append(dp); maxDz = max(maxDz, dz)
                    if same { topSame += 1 }
                    if (ca >= tau) == (cb >= tau) { routeSame += 1 }
                    rows.append(["clip": clipId, "prompt": pid, "p_int8": a.probabilities, "p_fp16": b.probabilities, "max_dp": dp, "max_dz": dz,
                                 "conf_int8": ca, "conf_fp16": cb, "ms_int8": a.msTotal, "ms_fp16": b.msTotal])
                } catch { skipped += 1 }
            }
        }
        if ProcessInfo.processInfo.arguments.contains("--fast-parity-cleanup"), fp16URL == docFP16 { try? FileManager.default.removeItem(at: docFP16) }
        let sorted = dps.sorted()
        let r: [String: Any] = ["n": dps.count, "skipped_too_long": skipped, "max_dp": sorted.last ?? -1,
                                "median_dp": sorted.isEmpty ? -1 : sorted[sorted.count / 2], "max_dz": maxDz,
                                "top_answer_same": topSame, "route_same_at_tau": routeSame, "tau": tau, "fp16_source": fp16URL == docFP16 ? "Documents" : "bundle",
                                "table": fast.table.format.rawValue, "rows": rows]
        Self.write(r, "fast_parity.json")
        status = ""
        return r
    }

    static func snapshot(_ name: String) {
        guard let window = UIApplication.shared.connectedScenes.compactMap({ ($0 as? UIWindowScene)?.keyWindow }).first else { return }
        let img = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in _ = window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
        try? img.pngData()?.write(to: URL.documentsDirectory.appendingPathComponent("snapshot_\(name).png"))
    }

    /// Free space for important (user-initiated) data on the app's volume, GB.
    static func diskFreeGB() -> Double {
        let v = try? URL.documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return Double(v?.volumeAvailableCapacityForImportantUsage ?? -1) / 1e9
    }

    static func json(_ o: Outcome) -> [String: Any] {
        var r: [String: Any] = ["id": o.questionId, "kind": o.kind.rawValue, "options": o.options, "multi": o.multi, "route": o.route.rawValue,
                                "tau": o.tau, "answer": o.answer, "answered_by": o.answeredBy, "memory": o.memory.dict, "note": o.note ?? ""]
        if let f = o.fast {
            r["fast_p"] = f.probabilities; r["fast_confidence"] = o.fastConfidence ?? -1
            r["fast_ms"] = ["audio": f.msAudio, "trunk": f.msTrunk, "total": f.msTotal]
            r["fast_z"] = f.logits; r["fast_diagnostics"] = f.diagnostics
            r["fast_tokens"] = f.tokens; r["fast_audio_tokens"] = f.audioTokens; r["fast_audio_s"] = f.audioSeconds
        }
        if let b = o.big {
            r["big_p"] = b.probabilities; r["big_z"] = b.logits ?? []; r["big_tokens"] = b.promptTokens
            r["big_ms"] = ["prefill": b.prefillMs, "readout": b.readoutMs, "load": o.bigLoadMs ?? 0]
        }
        r["fast_ref_torch"] = o.fastRef ?? []; r["big_ref"] = o.bigRef ?? []; r["big_build"] = o.bigBuild ?? ""
        return r
    }

    static func write(_ obj: [String: Any], _ name: String) {
        let url = URL.documentsDirectory.appendingPathComponent(name)
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) { try? data.write(to: url) }
    }

    // MARK: tokenizer check

    /// The audio model and Smaug-Decisions share the Gemma 4 BPE vocabulary and merges (tools/tokenizer_check.py); the audio model only adds the marker <unused0> (id 6,
    /// already in the shared vocabulary) as a special token and wraps prompts in <bos> ... <eos>. So the fast tier's ids can be
    /// built with the LiteRT-LM tokenizer: split the rendered prompt at the special pieces, tokenize the text between them.
    /// This rebuilds every pre-tokenised prompt that way and counts exact matches.
    func checkTokenizer() async {
        guard let file, let big else { tokenizerCheck = "skipped: Smaug-Decisions not loaded"; return }
        let prompts = file.scenarios.map { ($0.id, $0.fast) } + file.cases.compactMap { c in c.fast.map { (c.id, $0) } }
        let marker = file.marker_id, boa = file.audio_bos_id, eoa = file.audio_eos_id
        let result: String = (try? await onQueue(bigQueue) { () -> String in
            var ok = 0, bad: [String] = []
            for (id, p) in prompts {
                let (pre, post) = try FastTokenizer.ids(text: p.text, marker: marker, audioBOS: boa, audioEOS: eoa, tokenize: big.tokenize)
                if pre == p.pre_ids && post == p.post_ids { ok += 1 } else { bad.append(id) }
            }
            return "\(ok)/\(prompts.count) fast-tier prompts reproduced exactly by the LiteRT-LM tokenizer" + (bad.isEmpty ? "" : "; differ: \(bad.joined(separator: ","))")
        }) ?? "failed"
        tokenizerCheck = result
    }

    private func onQueue<T: Sendable>(_ q: DispatchQueue, _ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in q.async { cont.resume(with: Result(catching: work)) } }
    }
}
