import PhotosUI
import SwiftUI

/// The Ask screen: header with the device's models, first-run / preparing cards, the Ask card (audio, text, photo, examples),
/// then the results as cards with their technical details folded away. Styled after the ChatLLM iOS app (Design.swift).
struct ContentView: View {
    @EnvironmentObject var model: CascadeModel
    @EnvironmentObject var dl: ModelDownloader
    @State private var showModels = ProcessInfo.processInfo.arguments.contains("--show-models")

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 20) {
                        Header(showModels: $showModels).id("top")
                        if case .preparing(let start, let expected) = model.bigState { PreparingCard(start: start, expected: expected) }
                        if model.pendingDownload != nil || dl.phase.active { GetModelCard(showModels: $showModels) }
                        AskCard()
                        if let o = model.outcomes.first {
                            VStack(spacing: 10) { SectionTitle("Latest"); ResultCard(o: o) }.id("latest")
                        }
                        if model.outcomes.count > 1 {
                            VStack(spacing: 10) {
                                SectionTitle("Earlier")
                                ForEach(model.outcomes.dropFirst()) { ResultCard(o: $0, compact: true) }
                            }
                        }
                        if !model.status.isEmpty {
                            Text(model.status).font(.interCaption()).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4)
                        }
                    }
                    .padding(.horizontal, 20).padding(.top, 8).padding(.bottom, 40)
                }
                .scrollDismissesKeyboard(.interactively)
                .background(GradientBackground(isFlipped: !model.outcomes.isEmpty))
                .onChange(of: model.scroll) { _, r in withAnimation { proxy.scrollTo(r.target, anchor: .top) } }
            }
            .toolbar(.hidden, for: .navigationBar)
            .toolbar { ToolbarItemGroup(placement: .keyboard) { Spacer(); Button("Done") { hideKeyboard() }.tint(.violet) } }
            .sheet(isPresented: $showModels) { ModelsView().environmentObject(model).environmentObject(dl) }
            .task { await model.autorunFromLaunchArguments() }
        }
        .tint(.violet)
    }
}

func hideKeyboard() { UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil) }

func gb(_ mb: Double) -> String { mb >= 1024 ? String(format: "%.2f GB", mb / 1024) : String(format: "%.0f MB", mb) }

// MARK: header

struct Header: View {
    @EnvironmentObject var model: CascadeModel
    @EnvironmentObject var dl: ModelDownloader
    @Environment(\.colorScheme) private var colorScheme
    @Binding var showModels: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Cascade").font(.interTitle2(.bold)).lineLimit(1)
                    Text(subtitle).font(.interFootnote()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                }
                Spacer()
                Button { showModels = true } label: {
                    ZStack {
                        Circle().fill(colorScheme == .dark ? Color.white.opacity(0.08) : Color.white).frame(width: 40, height: 40)
                            .overlay(Circle().stroke(Color.primary.opacity(0.06)))
                        Lucide("sliders-horizontal", size: 18).foregroundStyle(.primary)
                    }
                }
                .buttonStyle(PressableScaleButtonStyle())
                .accessibilityLabel("Models")
            }
            HStack(spacing: 8) {
                ModelPill(name: "0.7B audio model", state: model.fastReady ? "ready" : "loading", dot: model.fastReady ? .green : .orange)
                ModelPill(name: "Smaug-Decisions", state: bigShort, dot: bigDot)
                Spacer(minLength: 0)
            }
        }
        .padding(.top, 8)
    }

    private var subtitle: String {
        guard let p = model.policy else { return "Two decision models, on this phone" }
        return (p.mode == .full ? model.choice.title : "Fast only") + " · on this phone"
    }

    /// Green when ready or loaded, amber while loading / preparing / downloading, grey when off or not installed.
    private var bigDot: Color {
        switch bigShort {
        case "ready", "loaded": return .green
        case "off", "not installed": return Color.secondary.opacity(0.4)
        default: return .orange
        }
    }

    private var bigShort: String {
        if model.policy?.mode == .fastOnly { return "off" }
        if !model.choice.usesBig { return "not installed" }
        if model.activeVariant == nil { return dl.phase.active ? "\(dl.percent)%" : "not installed" }
        switch model.bigState {
        case .loaded: return "loaded"
        case .loading: return "loading"
        case .preparing: return "preparing"
        default: return "ready"
        }
    }
}

struct ModelPill: View {
    @Environment(\.colorScheme) private var colorScheme
    let name: String, state: String, dot: Color
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(dot).frame(width: 7, height: 7)
            // one Text, so name and state shrink together: both pills share one row and never wrap or run off the screen
            (Text(name).font(.interCaption(.semibold)) + Text("  " + state).font(.interCaption()).foregroundStyle(.secondary))
                .lineLimit(1).minimumScaleFactor(0.6)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(Capsule().fill(colorScheme == .dark ? Color.white.opacity(0.06) : Color.white))
        .overlay(Capsule().stroke(Color.primary.opacity(0.06)))
    }
}

// MARK: ask

enum InputMode: String, CaseIterable, Identifiable {
    case audio = "Audio", text = "Text", photo = "Photo", examples = "Examples"
    var id: String { rawValue }
    var icon: String { switch self { case .audio: return "audio-lines"; case .text: return "file-text"; case .photo: return "image"; case .examples: return "sparkles" } }
}

struct AskCard: View {
    @EnvironmentObject var model: CascadeModel
    @Environment(\.colorScheme) private var colorScheme
    @StateObject private var rec = Recorder(maxSeconds: 10)
    @State private var mode = InputMode.examples
    @State private var scenarioId: String?
    @State private var text = ""
    @State private var question = ""
    @State private var options = "Yes, No"
    @State private var photo: PhotosPickerItem?
    @State private var photoURL: URL?
    @State private var thumb: UIImage?
    @State private var freeform = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Ask").font(.interTitle3(.bold))
            modePicker
            switch mode {
            case .audio: audio
            case .text: textInput
            case .photo: photoInput
            case .examples: examples
            }
        }
        .card(radius: 16, padding: 16)
        .onChange(of: photo) { Task { await loadPhoto() } }
        .onAppear { if scenarioId == nil { scenarioId = model.file?.scenarios.first?.id } }
    }

    /// Labels with icons when they fit, else labels only, else icons only: never wraps or overflows.
    private var modePicker: some View {
        ViewThatFits(in: .horizontal) { modeChips(icons: true, labels: true); modeChips(icons: false, labels: true); modeChips(icons: true, labels: false) }
    }

    private func modeChips(icons: Bool, labels: Bool) -> some View {
        HStack(spacing: 4) {
            ForEach(InputMode.allCases) { m in
                Button { withAnimation(.easeInOut(duration: 0.15)) { mode = m } } label: {
                    HStack(spacing: 5) {
                        if icons { Lucide(m.icon, size: 13) }
                        if labels { Text(m.rawValue).font(.interFootnote(.semibold)).lineLimit(1).fixedSize() }
                    }
                    .accessibilityLabel(m.rawValue)
                    .foregroundStyle(mode == m ? Color.white : Color.primary)
                    .padding(.horizontal, 8).padding(.vertical, 7)
                    .frame(minWidth: 40, maxWidth: .infinity)
                    .background(Capsule().fill(mode == m ? Color.violet : Color.clear))
                }
                .buttonStyle(PressableScaleButtonStyle(pressedScale: 0.95))
            }
        }
        .padding(3)
        .background(Capsule().fill(Color.primary.opacity(colorScheme == .dark ? 0.08 : 0.05)))
    }

    // audio
    private var audio: some View {
        VStack(alignment: .leading, spacing: 14) {
            if model.canTokenize {
                Toggle(isOn: $freeform) { Text("Ask your own question").font(.interSubheadline()).lineLimit(1) }.tint(.violet)
            }
            if freeform && model.canTokenize {
                questionFields
            } else {
                Menu {
                    Picker("Question", selection: $scenarioId) {
                        ForEach(model.file?.scenarios ?? []) { Text($0.title).tag(Optional($0.id)) }
                    }
                } label: {
                    FieldRow(label: "Question", value: scenario?.title ?? "Pick a question", trailingIcon: "chevron-down")
                }
                if let s = scenario { Text(s.question).font(.interCaption()).foregroundStyle(.secondary).lineLimit(2) }
            }
            HStack(spacing: 14) {
                Button { rec.isRecording ? rec.stop() : try? rec.start() } label: {
                    ZStack {
                        Circle().fill(rec.isRecording ? Color.red : Color.violet).frame(width: 56, height: 56)
                            .shadow(color: (rec.isRecording ? Color.red : Color.violet).opacity(0.3), radius: 8, y: 2)
                        Lucide(rec.isRecording ? "square" : "mic", size: 22).foregroundStyle(.white)
                    }
                }
                .buttonStyle(PressableScaleButtonStyle()).disabled(model.busy)
                VStack(alignment: .leading, spacing: 6) {
                    Text(rec.isRecording ? "Recording…" : rec.samples.isEmpty ? "Record up to 10 s" : "Clip ready")
                        .font(.interSubheadline(.semibold)).lineLimit(1)
                    ProgressTrack(value: min(rec.seconds, 10) / 10, color: rec.isRecording ? .red : .violet)
                    Text(String(format: "%.1f s", rec.seconds)).font(.interCaption().monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            GradientCapsuleButton(title: "Decide", icon: "sparkles",
                                  enabled: !(model.busy || rec.samples.isEmpty || (freeform && model.canTokenize ? question.isEmpty : scenario == nil))) {
                hideKeyboard(); rec.stop()
                let samples = rec.samples
                if freeform && model.canTokenize {
                    let opts = options.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                    let qText = question
                    Task { if let q = await model.question(recording: samples, question: qText, options: opts, multi: false) { await model.run(q) } }
                } else if let s = scenario {
                    let q = model.question(recording: samples, scenario: s)
                    Task { await model.run(q) }
                }
            }
        }
    }

    private var textInput: some View {
        VStack(alignment: .leading, spacing: 12) {
            InputField(placeholder: "Text to decide about", text: $text, lines: 2...5)
            questionFields
            GradientCapsuleButton(title: "Decide", icon: "sparkles", enabled: !(model.busy || text.isEmpty || question.isEmpty)) { runTyped(kind: .text) }
        }
    }

    private var photoInput: some View {
        VStack(alignment: .leading, spacing: 12) {
            PhotosPicker(selection: $photo, matching: .images) {
                HStack(spacing: 12) {
                    if let thumb {
                        Image(uiImage: thumb).resizable().scaledToFill().frame(width: 56, height: 56)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    } else { IconBadge(icon: "camera", size: 44) }
                    VStack(alignment: .leading, spacing: 2) {
                        Text(photoURL == nil ? "Choose a photo" : "Choose another photo").font(.interSubheadline(.semibold)).foregroundStyle(.primary).lineLimit(1)
                        Text("From your library").font(.interCaption()).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer()
                    Lucide("chevron-right", size: 16).foregroundStyle(.secondary)
                }
                .padding(12)
                .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(colorScheme == .dark ? 0.06 : 0.03)))
            }
            .buttonStyle(.plain)
            questionFields
            GradientCapsuleButton(title: "Decide", icon: "sparkles", enabled: !(model.busy || photoURL == nil || question.isEmpty)) { runTyped(kind: .image) }
        }
    }

    private var examples: some View {
        VStack(spacing: 6) {
            ForEach(model.file?.cases ?? []) { c in
                Button { model.runCase(c) } label: {
                    HStack(spacing: 12) {
                        IconBadge(icon: icon(c.kind), size: 32)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(c.title.replacingOccurrences(of: "Audio: ", with: "").replacingOccurrences(of: "Text: ", with: "").replacingOccurrences(of: "Image: ", with: "").capitalizedFirst)
                                .font(.interSubheadline(.semibold)).foregroundStyle(.primary).lineLimit(1).truncationMode(.tail)
                            Text(c.question).font(.interCaption()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                        }
                        Spacer(minLength: 4)
                        Lucide("chevron-right", size: 16).foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(colorScheme == .dark ? 0.04 : 0.02)))
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableScaleButtonStyle(pressedScale: 0.98))
                .disabled(model.busy || !model.fastReady)
                .opacity(model.busy || !model.fastReady ? 0.5 : 1)
            }
        }
    }

    private var questionFields: some View {
        VStack(spacing: 10) {
            InputField(placeholder: "Question", text: $question, lines: 1...3)
            InputField(placeholder: "Options, comma-separated", text: $options, lines: 1...1)
        }
    }

    private var scenario: FastScenario? { model.file?.scenarios.first { $0.id == scenarioId } }

    private func icon(_ k: Router.InputKind) -> String {
        switch k { case .audio: return "audio-lines"; case .text: return "file-text"; case .image: return "image" }
    }

    private func runTyped(kind: Router.InputKind) {
        hideKeyboard()
        let opts = options.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard (2...Prompt.letters.count).contains(opts.count) else { model.status = "Give 2 to 20 comma-separated options."; return }
        var q = Question(id: kind.rawValue, title: kind == .text ? "Text" : "Photo", kind: kind, question: question, options: opts,
                         descriptions: Array(repeating: nil, count: opts.count), multi: false, rules: nil)
        q.text = kind == .text ? text : nil
        q.imageURL = kind == .image ? photoURL : nil
        Task { await model.run(q) }
    }

    private func loadPhoto() async {
        guard let photo, let data = try? await photo.loadTransferable(type: Data.self), let img = UIImage(data: data) else { return }
        thumb = img
        photoURL = try? MediaPrep.png(from: img, name: "picked.png")
    }
}

extension String { var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() } }

struct InputField: View {
    @Environment(\.colorScheme) private var colorScheme
    let placeholder: String
    @Binding var text: String
    var lines: ClosedRange<Int> = 1...1
    var body: some View {
        TextField(placeholder, text: $text, axis: .vertical).lineLimit(lines).font(.interSubheadline())
            .submitLabel(.done)
            .padding(.horizontal, 14).padding(.vertical, 12)
            .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(colorScheme == .dark ? 0.06 : 0.035)))
    }
}

struct FieldRow: View {
    @Environment(\.colorScheme) private var colorScheme
    let label: String, value: String
    var trailingIcon: String? = nil
    var body: some View {
        HStack {
            Text(label).font(.interSubheadline()).foregroundStyle(.secondary).lineLimit(1).fixedSize()
            Spacer(minLength: 8)
            Text(value).font(.interSubheadline(.semibold)).foregroundStyle(.primary).lineLimit(1).truncationMode(.tail)
            if let trailingIcon { Lucide(trailingIcon, size: 14).foregroundStyle(.secondary) }
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(colorScheme == .dark ? 0.06 : 0.035)))
    }
}

// MARK: result

extension Router.Route {
    var label: String {
        switch self {
        case .fast: return "0.7B audio model"
        case .escalated: return "Escalated"
        case .routed: return "Smaug-Decisions"
        case .wouldEscalate: return "Would escalate"
        case .needsBig: return "Needs Smaug-Decisions"
        case .unsure: return "Unsure"
        case .notInstalled: return "Not installed"
        }
    }
    var color: Color {
        switch self {
        case .fast: return .green
        case .escalated: return .orange
        case .routed: return .violet
        case .wouldEscalate, .unsure: return .purple
        case .needsBig, .notInstalled: return .gray
        }
    }
    var icon: String {
        switch self {
        case .fast: return "zap"
        case .escalated: return "arrow-up"
        case .routed: return "brain"
        case .wouldEscalate, .unsure: return "circle-alert"
        case .needsBig, .notInstalled: return "box"
        }
    }
}

struct ResultCard: View {
    let o: Outcome
    var compact = false
    @State private var details = false
    @State private var expanded = false
    static let maxRows = 5

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 8) {
                Chip(text: o.route.label, icon: o.route.icon, color: o.route.color)
                Spacer(minLength: 4)
                Text(o.title.replacingOccurrences(of: "Audio: ", with: "").replacingOccurrences(of: "Text: ", with: "").replacingOccurrences(of: "Image: ", with: ""))
                    .font(.interCaption()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(headline).font(compact ? .interCallout(.semibold) : .interTitle3(.bold)).lineLimit(compact ? 1 : 2).truncationMode(.tail)
                if o.final != nil { Text("by \(o.answeredBy)").font(.interCaption()).foregroundStyle(.secondary).lineLimit(1) }
            }
            if let c = o.fastConfidence, !compact || expanded { ConfidenceBar(confidence: c, tau: o.tau) }
            if o.final != nil, !compact || expanded {
                ProbTable(o: o, rows: rows)
                if rows.count < o.options.count {
                    Text("+\(o.options.count - rows.count) more options").font(.interCaption2()).foregroundStyle(.tertiary)
                }
            }
            if let n = o.note, !n.isEmpty, o.route != .needsBig, o.route != .notInstalled {
                HStack(alignment: .top, spacing: 6) {
                    Lucide("info", size: 13).foregroundStyle(.orange).padding(.top, 1)
                    Text(n).font(.interCaption()).foregroundStyle(.secondary).lineLimit(2).truncationMode(.tail)
                }
            }
            Button { withAnimation(.easeInOut(duration: 0.15)) { if compact && !expanded { expanded = true } else { details.toggle() } } } label: {
                HStack(spacing: 4) {
                    Text(compact && !expanded ? "Show" : "Details").font(.interFootnote(.semibold)).lineLimit(1)
                    Lucide(details ? "chevron-down" : "chevron-right", size: 13)
                    Spacer()
                    StatChip(text: latencySummary, icon: "clock-4")
                }
                .foregroundStyle(Color.violet)
            }
            .buttonStyle(.plain)
            if details { DetailsBlock(o: o) }
        }
        .card(radius: 16, padding: 16)
    }

    private var headline: String {
        guard o.route == .needsBig || o.route == .notInstalled else { return o.answer }
        let n = o.note ?? o.answer
        return n.contains(" is free") ? "Not enough storage for Smaug-Decisions" : n
    }

    private var latencySummary: String {
        String(format: "%.2f s", o.totalMs / 1000)
    }

    private var rows: [Int] {
        let idx = Array(o.options.indices)
        guard o.options.count > Self.maxRows else { return idx }
        let score = { (i: Int) in max(o.fast?.probabilities[i] ?? 0, o.big?.probabilities[i] ?? 0) }
        return Array(idx.sorted { score($0) > score($1) }.prefix(Self.maxRows)).sorted()
    }
}

/// Latency, memory, tokens, references: the numbers behind a decision.
struct DetailsBlock: View {
    @Environment(\.colorScheme) private var colorScheme
    let o: Outcome
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let f = o.fast {
                DetailRow(k: "0.7B audio model", v: String(format: "%.0f ms · audio %.0f · trunk %.0f · %d tokens", f.msTotal, f.msAudio, f.msTrunk, f.tokens))
            }
            if let b = o.big {
                DetailRow(k: "Smaug-Decisions", v: String(format: "%.2f s · prefill %.2f · readout %.2f · %d tokens", (b.prefillMs + b.readoutMs) / 1000,
                                                  b.prefillMs / 1000, b.readoutMs / 1000, b.promptTokens))
                if let ms = o.bigLoadMs { DetailRow(k: "Load", v: String(format: "%.1f s", ms / 1000)) }
                if let eb = o.bigBuild { DetailRow(k: "Build", v: eb) }
            }
            DetailRow(k: "Memory", v: "\(gb(o.memory.footprintMB)) · peak \(gb(o.memory.footprintPeakMB))")
            if let n = o.note, !n.isEmpty { Text(n).font(.interCaption()).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
            if let d = o.detail, !d.isEmpty { Text(d).font(.interCaption2().monospaced()).foregroundStyle(.secondary).textSelection(.enabled).fixedSize(horizontal: false, vertical: true) }
            DetailRow(k: "τ", v: String(format: "%.2f", o.tau))
            if let r = o.fastRef { DetailRow(k: "0.7B audio model ref", v: r.map { String(format: "%.3f", $0) }.joined(separator: " ")) }
            if let r = o.bigRef { DetailRow(k: "Smaug-Decisions ref", v: r.map { String(format: "%.3f", $0) }.joined(separator: " ")) }
            if o.fastRef != nil || o.bigRef != nil { Text("ref = PyTorch reference for this input").font(.interCaption2()).foregroundStyle(.tertiary) }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.primary.opacity(colorScheme == .dark ? 0.06 : 0.03)))
    }
}

struct DetailRow: View {
    let k: String, v: String
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(k).font(.interCaption(.semibold)).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.7).frame(width: 118, alignment: .leading)
            Text(v).font(.interCaption().monospacedDigit()).foregroundStyle(.primary).lineLimit(1).minimumScaleFactor(0.7)
            Spacer(minLength: 0)
        }
    }
}

struct ConfidenceBar: View {
    let confidence: Double, tau: Double
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("0.7B audio model confidence").font(.interCaption()).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.8)
                Spacer(minLength: 6)
                Text(String(format: "%.2f %@ τ %.2f", confidence, confidence >= tau ? "≥" : "<", tau)).font(.interCaption(.semibold).monospacedDigit())
                    .foregroundStyle(confidence >= tau ? Color.green : Color.orange).lineLimit(1).fixedSize()
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule().fill(confidence >= tau ? Color.green : Color.orange).frame(width: max(4, g.size.width * confidence))
                    RoundedRectangle(cornerRadius: 1).fill(Color.primary.opacity(0.7)).frame(width: 2, height: 12).offset(x: g.size.width * tau - 1)
                }
            }
            .frame(height: 6)
        }
    }
}

struct ProbTable: View {
    let o: Outcome
    let rows: [Int]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(rows, id: \.self) { i in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(o.options[i]).font(.interFootnote(isTop(i) ? .semibold : .regular)).lineLimit(1).truncationMode(.tail)
                        Spacer(minLength: 6)
                        if let f = o.fast, o.big != nil { Text(String(format: "0.7B audio model %.2f", f.probabilities[i])).font(.interCaption2().monospacedDigit()).foregroundStyle(.secondary).lineLimit(1).fixedSize() }
                        Text(String(format: "%.3f", o.final?[i] ?? 0)).font(.interCaption(.semibold).monospacedDigit()).lineLimit(1).fixedSize()
                    }
                    ProgressTrack(value: o.final?[i] ?? 0, color: isTop(i) ? .violet : Color.secondary.opacity(0.6))
                }
            }
        }
    }
    private func isTop(_ i: Int) -> Bool {
        guard let p = o.final else { return false }
        return o.multi ? p[i] > 0.5 : p.indices.max { p[$0] < p[$1] } == i
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
