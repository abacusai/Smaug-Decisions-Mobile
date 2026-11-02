import SwiftUI

/// Settings › Models: what to install and use, the download, the cascade threshold, and this device. Sheet layout after the
/// ChatLLM app's HumanizePopupView: header badge + title, uppercased sections of selectable cards, gradient CTA at the bottom.
struct ModelsView: View {
    @EnvironmentObject var model: CascadeModel
    @EnvironmentObject var dl: ModelDownloader
    @Environment(\.dismiss) private var dismiss
    @State private var confirm: ModelPlan?

    var body: some View {
        ZStack {
            GradientBackground()
            VStack(spacing: 0) {
                HStack(spacing: 12) {
                    IconBadge(icon: "box", solid: true, size: 44)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Models").font(.interTitle3(.bold)).lineLimit(1)
                        Text("What runs on this iPhone").font(.interFootnote()).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 20).padding(.top, 24).padding(.bottom, 12)
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        VStack(spacing: 10) {
                            SectionTitle("What to install")
                            ForEach(model.offeredChoices) { c in
                                Button { choose(c) } label: { ChoiceRow(choice: c, selected: c == model.choice) }.buttonStyle(.plain)
                            }
                            if model.policy?.mode == .fastOnly {
                                Text("Not enough memory for Smaug-Decisions on this device")
                                    .font(.interCaption()).lineLimit(2).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4)
                            }
                        }
                        if model.pendingDownload != nil || dl.phase != .idle {
                            VStack(spacing: 10) { SectionTitle("Download"); DownloadPanel().card() }
                        }
                        if model.choice.escalatesAudio {
                            VStack(spacing: 10) {
                                SectionTitle("Cascade")
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack {
                                        Text("Escalate below").font(.interSubheadline(.semibold)).lineLimit(1)
                                        Spacer()
                                        Text(String(format: "τ %.2f", model.tau)).font(.interSubheadline(.semibold).monospacedDigit()).foregroundStyle(Color.violet)
                                    }
                                    Slider(value: $model.tau, in: 0.5...0.99, step: 0.01).tint(.violet)
                                    Text("The 0.7B audio model answers when its confidence ≥ τ").font(.interCaption()).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.75)
                                }
                                .card()
                            }
                        }
                        VStack(spacing: 10) { SectionTitle("This device"); DeviceCard().card(padding: 0) }
                        if let t = model.tokenizerCheck {
                            VStack(spacing: 10) { SectionTitle("Diagnostics"); Text(t).font(.interCaption()).foregroundStyle(.secondary).card() }
                        }
                    }
                    .padding(.horizontal, 20).padding(.bottom, 24)
                }
                GradientCapsuleButton(title: "Done") { dismiss() }.padding(.horizontal, 20).padding(.vertical, 12)
            }
        }
        .presentationDragIndicator(.visible)
        .confirmationDialog("Delete files you no longer need?", isPresented: Binding(get: { confirm != nil }, set: { if !$0 { confirm = nil } }),
                            titleVisibility: .visible, presenting: confirm) { p in
            Button("Delete and switch (frees \(ModelStorage.gb(p.freedBytes)))", role: .destructive) { model.apply(p.choice, deleteOthers: true) }
            Button("Switch, keep the files") { model.apply(p.choice, deleteOthers: false) }
            Button("Cancel", role: .cancel) {}
        } message: { p in
            Text(p.delete.map { "\($0.file) (\(ModelStorage.gb2($0.bytes)))" }.joined(separator: "\n"))
        }
    }

    private func choose(_ c: ModelChoice) {
        let p = model.plan(for: c)
        withAnimation(.easeInOut(duration: 0.15)) {
            if p.delete.isEmpty { model.apply(c, deleteOthers: false) } else { confirm = p }
        }
    }
}

extension ModelChoice {
    var icon: String {
        switch self { case .audioOnly: return "ear"; case .textImage: return "image"; case .audioText: return "audio-lines"; case .full: return "sparkles" }
    }
}

/// One choice with what it costs on this device.
struct ChoiceRow: View {
    @EnvironmentObject var model: CascadeModel
    let choice: ModelChoice, selected: Bool
    /// The device's default choice, marked on the first-run screen.
    var recommended = false

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            IconBadge(icon: choice.icon, solid: selected)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(choice.title).font(.interSubheadline(.semibold)).foregroundStyle(.primary).lineLimit(1).minimumScaleFactor(0.8).truncationMode(.tail)
                    if recommended { Chip(text: "Recommended") }
                }
                Text(choice.detail).font(.interCaption()).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.75).truncationMode(.tail)
                Text(cost).font(.interCaption2(.semibold).monospacedDigit()).foregroundStyle(selected ? Color.violet : .secondary).lineLimit(1).minimumScaleFactor(0.75)
            }
            Spacer(minLength: 8)
            RadioIndicator(selected: selected)
        }
        .card(radius: 14, padding: 12, selected: selected)
        .contentShape(Rectangle())
    }

    private var cost: String {
        guard choice.usesBig else { return "Nothing to download" }
        let p = model.plan(for: choice)
        guard let v = p.target else { return "Not in the manifest" }
        let est = model.manifest?.isEstimated(v) ?? false
        let size = (est ? "≈ " : "") + ModelStorage.gb2(v.bytes) + (est ? " (estimated)" : "")
        let storage = (est ? "≈ " : "") + ModelStorage.gb(v.bytes + v.cacheBytes)
        if p.download == nil { return "Installed · \(storage) on disk" }
        return "\(size) download · \(storage) on disk"
    }
}

/// The phone's mode, memory and storage, and the two models' state.
struct DeviceCard: View {
    @EnvironmentObject var model: CascadeModel
    @EnvironmentObject var dl: ModelDownloader

    var body: some View {
        VStack(spacing: 0) {
            if let p = model.policy {
                InfoRow(icon: "smartphone", title: p.mode.rawValue, detail: String(format: "%.0f GB RAM", p.physicalMB / 1024))
                Divider().padding(.leading, 56)
                InfoRow(icon: "gauge", title: "App memory", detail: String(format: "%.1f GB limit · %.1f GB free", p.limitMB / 1024, p.availableMB / 1024))
                Divider().padding(.leading, 56)
            }
            InfoRow(icon: "cpu", title: "In use", detail: "\(gb(model.memory.footprintMB)) · peak \(gb(model.memory.footprintPeakMB))")
            Divider().padding(.leading, 56)
            InfoRow(icon: "server", title: "Free storage", detail: ModelStorage.freeBytes().map(ModelStorage.gb) ?? "?")
            Divider().padding(.leading, 56)
            InfoRow(icon: "ear", title: "0.7B audio model", detail: model.fastState)
            Divider().padding(.leading, 56)
            HStack(spacing: 8) {
                InfoRow(icon: "brain", title: "Smaug-Decisions", detail: bigStateText)
                if !model.bigState.isLoaded, model.bigNeedsPrepare, model.policy?.mode == .full {
                    SecondaryCapsuleButton(title: "Prepare now", compact: true) { model.prepareNow() }.disabled(model.busy).padding(.trailing, 16)
                }
                if model.bigState.isLoaded {
                    SecondaryCapsuleButton(title: "Release", compact: true) { model.releaseBig("by hand") }.disabled(model.busy).padding(.trailing, 16)
                }
            }
            #if targetEnvironment(simulator)
            Text("Simulator: Mac CPU latencies")
                .font(.interCaption2()).foregroundStyle(.tertiary).lineLimit(1).padding(.horizontal, 16).padding(.bottom, 12)
            #endif
        }
    }

    private var bigStateText: String {
        if case .unavailable = model.bigState, model.activeVariant == nil, dl.phase.active {
            return dl.phase == .verifying ? "verifying the download…" : "downloading (\(dl.percent)%)"
        }
        return model.bigState.label
    }
}

struct InfoRow: View {
    let icon: String, title: String, detail: String
    var body: some View {
        HStack(spacing: 12) {
            Lucide(icon, size: 18).foregroundStyle(.secondary).frame(width: 28)
            Text(title).font(.interSubheadline()).lineLimit(1).fixedSize()
            Spacer(minLength: 8)
            Text(detail).font(.interFootnote().monospacedDigit()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail).minimumScaleFactor(0.8)
        }
        .padding(.horizontal, 16).padding(.vertical, 13)
    }
}

/// The download itself: size, storage needed, Wi-Fi advice, progress, pause / resume, errors.
struct DownloadPanel: View {
    @EnvironmentObject var model: CascadeModel
    @EnvironmentObject var dl: ModelDownloader

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch dl.phase {
            case .downloading:
                HStack {
                    Text(dl.waitingForWifi ? "Waiting for Wi-Fi" : "Downloading").font(.interSubheadline(.semibold)).lineLimit(1)
                    Spacer()
                    Text("\(dl.percent)%").font(.interSubheadline(.semibold).monospacedDigit()).foregroundStyle(Color.violet)
                }
                ProgressTrack(value: dl.fraction)
                HStack {
                    Text("\(ModelStorage.gb2(dl.received)) of \(ModelStorage.gb2(dl.item?.bytes ?? 0))").font(.interCaption().monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                    Spacer()
                    SecondaryCapsuleButton(title: "Pause", icon: "pause", compact: true) { dl.pause() }
                }
                Text(dl.waitingForWifi ? "Starts on Wi-Fi · continues in the background" : "Continues in the background").font(.interCaption()).foregroundStyle(.secondary).lineLimit(1)
            case .verifying:
                Text("Checking the file").font(.interSubheadline(.semibold)).lineLimit(1)
                ProgressTrack(value: dl.verifyFraction)
                Text("sha256").font(.interCaption()).foregroundStyle(.secondary).lineLimit(1)
            case .paused(let why), .failed(let why):
                HStack(alignment: .top, spacing: 8) {
                    Lucide("circle-alert", size: 16).foregroundStyle(.orange)
                    Text(why).font(.interFootnote()).foregroundStyle(.primary).lineLimit(2).truncationMode(.tail)
                }
                if model.pendingDownload != nil {
                    GradientCapsuleButton(title: dl.item.map { dl.resumable($0.file) } ?? false ? "Resume" : "Try again", icon: "rotate-cw", enabled: !blocked) { model.startDownload() }
                }
            case .done:
                HStack(spacing: 8) {
                    Lucide("check", size: 16).foregroundStyle(.green)
                    Text("Downloaded and verified").font(.interSubheadline(.semibold)).lineLimit(1)
                }
            case .idle:
                if let v = model.pendingDownload {
                    let est = model.manifest?.isEstimated(v) ?? false
                    VStack(spacing: 8) {
                        StatLine(icon: "circle-arrow-down", label: "Download", value: (est ? "≈ " : "") + ModelStorage.gb2(v.bytes))
                        StatLine(icon: "server", label: "Needs free", value: ModelStorage.gb(ModelStorage.downloadNeed(v)))
                        StatLine(icon: "info", label: "Free now", value: ModelStorage.freeBytes().map(ModelStorage.gb) ?? "?")
                        StatLine(icon: "cpu", label: "One-time cache", value: ModelStorage.gb(v.cacheBytes))
                    }
                    HStack(spacing: 6) {
                        Image(systemName: dl.onCellular ? "antenna.radiowaves.left.and.right" : "wifi").font(.system(size: 12, weight: .semibold))
                        Text(dl.onCellular ? "On cellular · Wi-Fi recommended" : "Wi-Fi recommended").font(.interCaption()).lineLimit(1)
                    }
                    .foregroundStyle(dl.onCellular ? Color.orange : .secondary)
                    if let why = model.downloadBlockerShort(v) {
                        HStack(spacing: 8) {
                            Lucide("triangle-alert", size: 15).foregroundStyle(.orange)
                            Text(why).font(.interFootnote(.semibold)).lineLimit(1)
                        }
                    }
                    GradientCapsuleButton(title: dl.onCellular ? "Download on cellular" : "Download", icon: "circle-arrow-down", enabled: !blocked) { model.startDownload() }
                }
            }
        }
    }

    private var blocked: Bool { model.pendingDownload.flatMap(model.downloadBlocker) != nil }
}

struct StatLine: View {
    let icon: String, label: String, value: String
    var body: some View {
        HStack(spacing: 10) {
            Lucide(icon, size: 15).foregroundStyle(.secondary)
            Text(label).font(.interFootnote()).foregroundStyle(.secondary).lineLimit(1)
            Spacer(minLength: 8)
            Text(value).font(.interFootnote(.semibold).monospacedDigit()).lineLimit(1).fixedSize()
        }
    }
}

/// The first-run card: shown while the chosen models need a Smaug-Decisions file that is not on the device.
struct GetModelCard: View {
    @EnvironmentObject var model: CascadeModel
    @Binding var showModels: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                IconBadge(icon: "circle-arrow-down", solid: true, size: 44)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Get Smaug-Decisions").font(.interTitle3(.bold)).lineLimit(1)
                    Text("Text, photos, unsure audio").font(.interFootnote()).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            Button { showModels = true } label: { FieldRow(label: "Use", value: model.choice.title, trailingIcon: "chevron-right") }.buttonStyle(.plain)
            DownloadPanel()
        }
        .card()
    }
}

/// The one-time first load of Smaug-Decisions (its weight cache is written): elapsed against the expected time.
struct PreparingCard: View {
    let start: Date, expected: Double
    var body: some View {
        TimelineView(.periodic(from: start, by: 0.5)) { ctx in
            let t = ctx.date.timeIntervalSince(start)
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    IconBadge(icon: "cpu", solid: true, size: 44)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Preparing Smaug-Decisions").font(.interTitle3(.bold)).lineLimit(1)
                        Text(t < expected ? String(format: "%.0f s of about %.0f s", t, expected) : String(format: "%.0f s · almost done", t))
                            .font(.interFootnote().monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                ProgressTrack(value: min(t / expected, 0.97))
                Text("One-time setup · later loads take under a second")
                    .font(.interCaption()).foregroundStyle(.secondary).lineLimit(2)
            }
            .card()
        }
    }
}
