import CryptoKit
import Foundation
import Network
import UIKit

/// Downloads a Smaug-Decisions variant in the background: a URLSession background configuration (keeps going while the app is suspended,
/// and iOS relaunches the app to deliver the result), resumable (resume data is kept on disk after a pause or a dropped
/// connection), sha256-verified, then moved into Documents/<file> and excluded from backup.
@MainActor
final class ModelDownloader: NSObject, ObservableObject {
    static let shared = ModelDownloader()
    static let sessionId = "ai.abacus.cascade.model-download"

    /// What a download needs; stored in the task's description so a relaunch knows what the task is for.
    struct Item: Codable, Equatable, Sendable {
        let file: String, url: String, sha256: String, bytes: Int64
        /// The first-run "Download on Wi-Fi only" toggle: the task's request does not use cellular (nil in items from older builds).
        var wifiOnly: Bool? = nil
    }

    enum Phase: Equatable {
        case idle
        case downloading
        case paused(String)
        case verifying
        case done(String)          // file name
        case failed(String)
        var active: Bool { self == .downloading || self == .verifying }
    }

    @Published private(set) var phase = Phase.idle
    @Published private(set) var item: Item?
    @Published private(set) var received: Int64 = 0
    @Published private(set) var verifyFraction = 0.0
    @Published private(set) var onCellular = false
    @Published private(set) var startedAt: Date?
    /// The running download waits for Wi-Fi while the phone is on cellular.
    var waitingForWifi: Bool { phase == .downloading && (item?.wifiOnly ?? false) && onCellular }
    var fraction: Double { guard let item, item.bytes > 0 else { return 0 } ; return min(1, Double(received) / Double(item.bytes)) }
    var percent: Int { Int((phase == .verifying ? 1 : fraction) * 100) }
    /// Called on the main actor once a verified file is in place.
    var onInstalled: ((String) -> Void)?
    /// Set by the app delegate when iOS relaunches the app for this session's events.
    var backgroundCompletion: (() -> Void)?

    private var session: URLSession!
    private var task: URLSessionDownloadTask?
    private var pausing = false
    private var lastPublish = Date.distantPast
    private let monitor = NWPathMonitor()

    override private init() {
        super.init()
        let cfg = URLSessionConfiguration.background(withIdentifier: Self.sessionId)
        cfg.isDiscretionary = false            // start now (the user asked), not when iOS finds it convenient
        cfg.sessionSendsLaunchEvents = true
        cfg.allowsCellularAccess = true        // the card recommends Wi-Fi and warns on cellular; a Wi-Fi-only item's request opts out
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
        monitor.pathUpdateHandler = { [weak self] path in
            let cell = path.usesInterfaceType(.cellular) && !path.usesInterfaceType(.wifi)
            Task { @MainActor in self?.onCellular = cell }
        }
        monitor.start(queue: DispatchQueue(label: "ai.abacus.cascade.path"))
        // a task that kept running while the app was not (or a finished file that still needs verifying)
        session.getAllTasks { tasks in
            let t = tasks.compactMap { $0 as? URLSessionDownloadTask }.first { $0.state == .running || $0.state == .suspended }
            Task { @MainActor in
                if let t, let item = Self.decode(t.taskDescription) {
                    self.task = t; self.item = item; self.received = t.countOfBytesReceived; self.phase = .downloading; self.startedAt = Date()
                } else { self.resumePendingVerification() }
            }
        }
    }

    // MARK: files

    nonisolated static var stagingDir: URL {
        let d = URL.applicationSupportDirectory.appendingPathComponent("model-download", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        var u = d; var v = URLResourceValues(); v.isExcludedFromBackup = true; try? u.setResourceValues(v)
        return d
    }
    nonisolated static func stagedURL(_ file: String) -> URL { stagingDir.appendingPathComponent(file + ".download") }
    nonisolated static func resumeURL(_ file: String) -> URL { stagingDir.appendingPathComponent(file + ".resume") }
    nonisolated static func itemURL(_ file: String) -> URL { stagingDir.appendingPathComponent(file + ".item.json") }
    nonisolated static func finalURL(_ file: String) -> URL { URL.documentsDirectory.appendingPathComponent(file) }

    nonisolated static func decode(_ s: String?) -> Item? { s.flatMap { try? JSONDecoder().decode(Item.self, from: Data($0.utf8)) } }
    nonisolated static func encode(_ i: Item) -> String { String(decoding: (try? JSONEncoder().encode(i)) ?? Data(), as: UTF8.self) }

    /// Bytes already on disk for a paused download of this file (resume data does not say; this is what was last reported).
    func resumable(_ file: String) -> Bool { FileManager.default.fileExists(atPath: Self.resumeURL(file).path) }

    // MARK: control

    func start(_ i: Item) {
        guard task == nil, !phase.active else { return }
        guard let url = URL(string: i.url) else { phase = .failed("Bad download URL in the manifest."); return }
        item = i; received = 0; startedAt = Date()
        try? Data(Self.encode(i).utf8).write(to: Self.itemURL(i.file))
        let t: URLSessionDownloadTask
        if let rd = try? Data(contentsOf: Self.resumeURL(i.file)) {
            t = session.downloadTask(withResumeData: rd)
            received = UserDefaults.standard.object(forKey: "dl.received.\(i.file)") as? Int64 ?? 0
        } else {
            var req = URLRequest(url: url)
            req.allowsCellularAccess = !(i.wifiOnly ?? false)
            t = session.downloadTask(with: req)
        }
        t.taskDescription = Self.encode(i)
        t.countOfBytesClientExpectsToReceive = i.bytes
        task = t; phase = .downloading
        t.resume()
    }

    func pause() {
        guard let t = task, let i = item else { return }
        pausing = true
        t.cancel { data in
            Task { @MainActor in
                if let data { try? data.write(to: Self.resumeURL(i.file)) }
                self.task = nil; self.pausing = false
                self.phase = .paused("Paused at \(ModelStorage.gb2(self.received)) of \(ModelStorage.gb2(i.bytes)).")
            }
        }
    }

    /// Cancels and forgets a download (switching choices).
    func cancel() {
        task?.cancel(); task = nil
        if let i = item { for u in [Self.resumeURL(i.file), Self.stagedURL(i.file), Self.itemURL(i.file)] { try? FileManager.default.removeItem(at: u) } }
        item = nil; received = 0; phase = .idle
    }

    func reset() { if case .done = phase { phase = .idle }; if case .failed = phase { phase = .idle } }

    // MARK: progress and completion (called from the delegate)

    fileprivate func progress(_ written: Int64, _ expected: Int64) {
        received = written
        if Date().timeIntervalSince(lastPublish) > 0.25 || written == expected {
            lastPublish = Date()
            if let i = item { UserDefaults.standard.set(written, forKey: "dl.received.\(i.file)") }
            if phase != .downloading, !pausing { phase = .downloading }
        }
    }

    fileprivate func finished(item i: Item?, staged: URL?, error: String?) {
        task = nil
        if let error { phase = .failed(error); return }
        guard let i, let staged else { phase = .failed("Download finished without a file."); return }
        item = i
        verify(staged, i)
    }

    fileprivate func interrupted(_ i: Item?, resumeData: Data?, message: String) {
        guard !pausing, let i, item?.file == i.file else { return }   // a pause or a cancel() already handled it
        if case .paused = phase { return }
        task = nil
        if let resumeData { try? resumeData.write(to: Self.resumeURL(i.file)); phase = .paused("\(message) It resumes where it stopped.") }
        else { phase = .failed(message) }
    }

    /// A staged file whose verification did not finish (the app was killed while hashing).
    private func resumePendingVerification() {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: Self.stagingDir.path) else { return }
        for n in names where n.hasSuffix(".item.json") {
            let file = String(n.dropLast(".item.json".count))
            guard let data = try? Data(contentsOf: Self.itemURL(file)), let i = try? JSONDecoder().decode(Item.self, from: data) else { continue }
            if FileManager.default.fileExists(atPath: Self.stagedURL(file).path) { item = i; verify(Self.stagedURL(file), i); return }
            if resumable(file) { item = i; received = UserDefaults.standard.object(forKey: "dl.received.\(file)") as? Int64 ?? 0
                phase = .paused("Download paused at \(ModelStorage.gb2(received)) of \(ModelStorage.gb2(i.bytes))."); return }
        }
    }

    private func verify(_ staged: URL, _ i: Item) {
        phase = .verifying; verifyFraction = 0
        let bg = UIApplication.shared.beginBackgroundTask(withName: "verify Smaug-Decisions")   // hashing 3.45 GB takes a few seconds
        Task.detached(priority: .userInitiated) {
            let result: Result<URL, Error> = Result {
                let (hex, n) = try Self.sha256(of: staged) { f in Task { @MainActor in self.verifyFraction = f } }
                guard n == i.bytes else { throw DeciderError(message: "The download has \(n) bytes, the manifest says \(i.bytes).") }
                guard hex == i.sha256.lowercased() else { throw DeciderError(message: "Checksum mismatch: the download was corrupted.") }
                let dst = Self.finalURL(i.file)
                try? FileManager.default.removeItem(at: dst)
                try FileManager.default.moveItem(at: staged, to: dst)
                var u = dst; var v = URLResourceValues(); v.isExcludedFromBackup = true; try u.setResourceValues(v)   // 3.45 GB: never to iCloud
                return dst
            }
            await MainActor.run {
                for u in [Self.resumeURL(i.file), Self.itemURL(i.file)] { try? FileManager.default.removeItem(at: u) }
                switch result {
                case .success:
                    self.phase = .done(i.file); self.onInstalled?(i.file)
                case .failure(let e):
                    try? FileManager.default.removeItem(at: staged)
                    self.phase = .failed("\(e.localizedDescription) It was deleted; try again.")
                }
                UIApplication.shared.endBackgroundTask(bg)
            }
        }
    }

    nonisolated static func sha256(of url: URL, progress: (Double) -> Void) throws -> (String, Int64) {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        let total = max(1, (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 1)
        var hasher = SHA256(), n: Int64 = 0, lastReport: Int64 = 0
        while true {
            let more: Bool = try autoreleasepool {
                guard let chunk = try h.read(upToCount: 8 << 20), !chunk.isEmpty else { return false }
                hasher.update(data: chunk); n += Int64(chunk.count)
                return true
            }
            if !more { break }
            if n - lastReport > 100_000_000 { lastReport = n; progress(Double(n) / Double(total)) }
        }
        return (hasher.finalize().map { String(format: "%02x", $0) }.joined(), n)
    }
}

extension ModelDownloader: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        Task { @MainActor in self.progress(totalBytesWritten, totalBytesExpectedToWrite) }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didResumeAtOffset fileOffset: Int64, expectedTotalBytes: Int64) {
        Task { @MainActor in self.progress(fileOffset, expectedTotalBytes) }
    }

    /// The file at `location` is deleted when this returns, so it is moved to the staging folder here (same volume: a rename).
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let item = Self.decode(downloadTask.taskDescription)
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status), let item else {
            let why = status == 401 || status == 403
                ? "The download server refused (HTTP \(status)). Copy the file into the app's Documents with Finder or devicectl."
                : "The download failed (HTTP \(status))."
            Task { @MainActor in self.finished(item: item, staged: nil, error: why) }
            return
        }
        let staged = Self.stagedURL(item.file)
        try? FileManager.default.removeItem(at: staged)
        do {
            try FileManager.default.moveItem(at: location, to: staged)
            Task { @MainActor in self.finished(item: item, staged: staged, error: nil) }
        } catch {
            Task { @MainActor in self.finished(item: item, staged: nil, error: "Could not keep the downloaded file: \(error.localizedDescription)") }
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }   // success went through didFinishDownloadingTo
        let ns = error as NSError
        let rd = ns.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let item = Self.decode(task.taskDescription)
        let msg = ns.code == NSURLErrorCancelled ? "Download stopped." : "Download interrupted (\(ns.localizedDescription))."
        Task { @MainActor in self.interrupted(item, resumeData: rd, message: msg) }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in self.backgroundCompletion?(); self.backgroundCompletion = nil }
    }
}

/// Hands iOS's background-session relaunch to the downloader.
final class CascadeAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String, completionHandler: @escaping () -> Void) {
        guard identifier == ModelDownloader.sessionId else { completionHandler(); return }
        Task { @MainActor in ModelDownloader.shared.backgroundCompletion = completionHandler }
    }
}
