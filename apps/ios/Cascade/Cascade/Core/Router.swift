import Foundation

/// The cascade's routing rule, ported from smaug_decide/cascade.py (confidence) and docs/ios.md (routing).
enum Router {
    static let defaultTau = 0.9

    /// smaug_decide.cascade.confidence: max(p) / max(sum(p), 1e-9) for one-answer questions; for select-all questions the least
    /// certain option's max(p, 1 - p).
    static func confidence(_ p: [Double], multi: Bool) -> Double {
        if multi { return p.map { max($0, 1 - $0) }.min() ?? 1 }
        return (p.max() ?? 0) / max(p.reduce(0, +), 1e-9)
    }

    enum InputKind: String, Codable, Sendable { case audio, text, image }

    /// Where a question goes. Audio: the fast tier first, escalate to Smaug-Decisions below tau. Text / image: straight to Smaug-Decisions ("routed"; the
    /// fast tier is near chance on text and would escalate 98% of it at 0.9). Without Smaug-Decisions on the device (Fast-only mode, or not
    /// enough memory left to load it) an unsure audio decision is kept and flagged, and text / image cannot be answered.
    enum Route: String, Codable, Sendable {
        case fast            // fast tier confident: its answer stands
        case escalated       // fast tier unsure: Smaug-Decisions answered
        case routed          // text or image: Smaug-Decisions answered directly
        case wouldEscalate = "would_escalate"   // fast tier unsure, Smaug-Decisions not available: fast answer flagged
        case needsBig = "needs_big"             // text or image, Smaug-Decisions not available
        case unsure                             // fast tier unsure and the chosen models do not escalate audio: flagged
        case notInstalled = "not_installed"     // the chosen models do not include this kind (e.g. images not installed)
    }

    /// The first step for an input kind: does it start on the fast tier?
    static func startsOnFast(_ kind: InputKind) -> Bool { kind == .audio }

    /// The route after the fast tier ran (audio) or for text/image, given whether Smaug-Decisions can be used.
    /// `choice` (Settings › Models) decides what Smaug-Decisions is for: a kind it does not cover is never sent to Smaug-Decisions. Unsure audio
    /// without audio in Smaug-Decisions stays with the fast tier as "unsure"; text or images Smaug-Decisions was not installed for are
    /// "not installed".
    static func route(kind: InputKind, fastConfidence: Double?, tau: Double, bigAvailable: Bool, choice: ModelChoice = .full) -> Route {
        if kind != .audio {
            guard choice.answers(kind) else { return .notInstalled }
            return bigAvailable ? .routed : .needsBig
        }
        if let c = fastConfidence, c >= tau { return .fast }
        guard choice.escalatesAudio else { return .unsure }
        return bigAvailable ? .escalated : .wouldEscalate
    }
}
