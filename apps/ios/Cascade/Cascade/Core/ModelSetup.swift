import Foundation

/// Free storage and what Smaug-Decisions needs of it.
enum ModelStorage {
    /// Left free for iOS itself after Smaug-Decisions' files: a phone at 0 bytes free misbehaves.
    static let headroomBytes: Int64 = 500_000_000

    /// Free space for user-initiated data on the app's volume (iOS counts purgeable space here), bytes; nil if unknown.
    static func freeBytes() -> Int64? {
        // --fake-free-gb x: pretend the volume has x GB free (to see the storage messages without filling the phone)
        let a = ProcessInfo.processInfo.arguments
        if let i = a.firstIndex(of: "--fake-free-gb"), i + 1 < a.count, let g = Double(a[i + 1]) { return Int64(g * 1e9) }
        let v = try? URL.documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage
    }

    /// Bytes the download still needs: the rest of the file, its whole weight cache, and the headroom. Space freed by deleting
    /// other variants (after confirmation) counts toward it.
    static func downloadNeed(_ v: ModelManifest.Variant, alreadyDownloaded: Int64 = 0) -> Int64 {
        max(0, v.bytes - alreadyDownloaded) + v.cacheBytes + headroomBytes
    }

    /// Bytes the first load still needs: the part of the weight cache not written yet, and the headroom.
    static func firstLoadNeed(_ v: ModelManifest.Variant, existingCache: Int64, vision: Bool) -> Int64 {
        max(0, (vision ? v.cacheBytes : v.cacheBytesWithoutVision) - existingCache) + headroomBytes
    }

    static func gb(_ b: Int64) -> String { String(format: "%.1f GB", Double(b) / 1e9) }
    static func gb2(_ b: Int64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

    /// "Free up 2.2 GB" (the short form for the UI), or nil when it fits.
    static func shortShortfall(need: Int64, free: Int64?) -> String? {
        guard let free, free < need else { return nil }
        return "Free up \(gb(need - free)) of storage"
    }

    /// "<what> needs 7.9 GB free and 2.1 GB is free. Free up 5.8 GB …", or nil when it fits.
    static func shortfall(need: Int64, free: Int64?, what: String) -> String? {
        guard let free, free < need else { return nil }
        return "\(what) needs \(gb(need)) free and \(gb(free)) is free. Free up \(gb(need - free)) in Settings › General › iPhone Storage, then try again."
    }
}

/// The one-time "Preparing Smaug-Decisions" step: LiteRT-LM writes its weight cache on the first load of a model file (13-21 s on the
/// iPhone 14 Pro Max, GPU backend; 8-10 s in the Simulator). Done is persisted per file, size and backend, and only counts while
/// that file's cache is still on disk (`--clear-cache`, or deleting the variant, makes the next load a first load again).
enum ModelPrep {
    static func key(file: String, bytes: Int64, backend: String) -> String { "smaug.prepared.\(file).\(bytes).\(backend)" }

    static func isPrepared(file: String, bytes: Int64, backend: String) -> Bool {
        let flag = UserDefaults.standard.bool(forKey: key(file: file, bytes: bytes, backend: backend))
        let cache = existingCache(file: file, flagged: flag)
        if cache < min(1_000_000_000, bytes / 4) { return false }   // no decoder cache for this file yet
        // the flag, or a full cache from a build before the flag existed
        return flag || cache >= 2_000_000_000
    }

    /// The cache already written for a file: its own named files, plus (once its first load is recorded) cache files that name
    /// no model file, in case a backend names its cache differently.
    static func existingCache(file: String, flagged: Bool) -> Int64 {
        BigEngine.cacheBytes(model: file) + (flagged ? BigEngine.unattributedCacheBytes() : 0)
    }

    static func markPrepared(file: String, bytes: Int64, backend: String, seconds: Double) {
        UserDefaults.standard.set(true, forKey: key(file: file, bytes: bytes, backend: backend))
        UserDefaults.standard.set(seconds * 1e9 / Double(max(bytes, 1)), forKey: "smaug.prepare_seconds_per_gb.\(backend)")
    }

    static func clearAll() {
        for k in UserDefaults.standard.dictionaryRepresentation().keys where k.hasPrefix("smaug.prepared.") { UserDefaults.standard.removeObject(forKey: k) }
    }

    /// What the progress bar counts toward: the last prepare time measured on this device for this backend, scaled to the file
    /// size; else 6.5 s per GB on a phone (2.5 s per GB in the Simulator).
    static func expectedSeconds(fileBytes: Int64, backend: String) -> Double {
        #if targetEnvironment(simulator)
        let base = 2.5
        #else
        let base = 6.5
        #endif
        let measured = UserDefaults.standard.double(forKey: "smaug.prepare_seconds_per_gb.\(backend)")
        return max(5, (measured > 0.5 ? measured : base) * Double(fileBytes) / 1e9)
    }
}
