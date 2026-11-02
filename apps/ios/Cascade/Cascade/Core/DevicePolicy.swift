import Foundation

/// Which cascade a device can run, and with which Smaug-Decisions build.
///
/// - The build follows the RAM class: 8 GB and up get `vis560` (images at 560 soft tokens, as trained); 6 GB phones get `vis280`
///   (the same file except a 280-token vision encoder: an image costs about 0.26 GB instead of 1.2 GB); smaller phones get none.
///   Both builds have a 2048-token KV cache. The vision encoder is attached only when an image arrives.
/// - The mode follows the memory still available to the app with the fast tier loaded: os_proc_available_memory() on a phone (it
///   reflects the increased-memory-limit entitlement when the OS grants it). Where that call returns 0 (the Simulator) the limit
///   is estimated from the RAM size (`--ram-gb n` stands in for an n GB phone).
/// - Smaug-Decisions loads only when the memory available covers `loadNeedMB`; an image additionally needs `visionNeedMB` at that moment.
///   Both are footprints measured in this app (Simulator and an iPhone 14 Pro Max, docs/devices.md) plus 25%.
struct DevicePolicy: Sendable {
    enum Mode: String, Codable, Sendable {
        case full = "Full cascade"
        case fastOnly = "Fast only"
    }

    enum Build: String, Codable, Sendable, CaseIterable {
        case vis560, vis280
        var file: String { "smaug-decisions-full-\(rawValue).litertlm" }
        var visualTokens: Int { self == .vis560 ? 560 : 280 }
        /// Footprint Smaug-Decisions adds over the fast tier for a text or audio decision (engine without vision), plus 25%. Measured
        /// (docs/devices.md): up to 0.83 GB in the Simulator (first load, while the weight cache is written) and 0.82 GB on the
        /// iPhone 14 Pro Max (CPU backend; 0.58 GB on its GPU backend). 0.85 GB x 1.25 ~ 1.1 GB, the same for both builds.
        var loadNeedMB: Double { 1100 }
        /// Footprint attaching the vision encoder and running one image adds on top of that, plus 25%. Measured: vis560 +1.13 GB
        /// (Simulator), vis280 +0.15 GB (Simulator) and up to +0.42 GB (iPhone 14 Pro Max).
        var visionNeedMB: Double { self == .vis560 ? 1450 : 550 }
    }

    /// Jetsam limits measured by developers where os_proc_available_memory is not available (no official Apple table exists):
    /// 8 GB and 12 GB iPhones with increased-memory-limit: about 6144 MB (Apple Developer Forums thread 834805, iPhone 16 Pro Max
    /// and 17 Pro Max); 6 GB: 3054 MB without the entitlement (iPhone 12 Pro, Stack Overflow q 5887248); 4 GB: about 2068-2098 MB
    /// (iPhone 11, same answer; Developer Forums thread 688973); smaller: half the RAM.
    static func estimatedLimitMB(physicalMB: Double) -> Double {
        let gb = physicalMB / 1024
        if gb >= 7.5 { return 6144 }
        if gb >= 5.5 { return 3054 }
        if gb >= 3.5 { return 2068 }
        return physicalMB * 0.5
    }

    /// The build for a RAM class: >= 8 GB (phones report a little under 8) vis560, 6 GB vis280, else none. Under 6 GB the Smaug-Decisions
    /// working set (about 3 GB of mostly clean, hot pages next to the OS) does not fit even when the dirty footprint would.
    static func build(physicalMB: Double) -> Build? {
        let gb = physicalMB / 1024
        if gb >= 7.5 { return .vis560 }
        if gb >= 5.5 { return .vis280 }
        return nil
    }

    static func mode(build: Build?, availableMB: Double) -> Mode {
        guard let build else { return .fastOnly }
        return availableMB >= build.loadNeedMB ? .full : .fastOnly
    }

    let physicalMB: Double
    let limitMB: Double
    let availableMB: Double
    let limitSource: String
    let build: Build?
    let mode: Mode
    let forced: Bool

    /// `ramOverrideGB` (--ram-gb) pretends the device has that much RAM and uses the estimate even on a phone; `forcedMode`
    /// (--mode fast-only|full) and `forcedBuild` (--build vis560|vis280) skip those decisions.
    static func evaluate(memory: MemorySnapshot, ramOverrideGB: Double? = nil, forcedMode: Mode? = nil, forcedBuild: Build? = nil) -> DevicePolicy {
        let physical = ramOverrideGB.map { $0 * 1024 } ?? Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576
        let limit: Double, avail: Double, source: String
        if ramOverrideGB == nil, let a = memory.availableMB {
            avail = a; limit = a + memory.footprintMB; source = "os_proc_available_memory"
        } else {
            limit = estimatedLimitMB(physicalMB: physical); avail = max(0, limit - memory.footprintMB)
            source = ramOverrideGB == nil ? "estimated from RAM" : "estimated for a \(Int(ramOverrideGB!)) GB phone"
        }
        let b = forcedBuild ?? build(physicalMB: physical) ?? (forcedMode == .full ? .vis560 : nil)
        return DevicePolicy(physicalMB: physical, limitMB: limit, availableMB: avail, limitSource: source, build: b,
                            mode: forcedMode ?? mode(build: b, availableMB: avail), forced: forcedMode != nil)
    }
}
