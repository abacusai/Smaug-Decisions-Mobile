import SwiftUI

/// First launch: "Choose your setup", before the home screen and before anything downloads. The rows are the Models screen's
/// (what this device is offered, with download size and storage), the device's default marked "Recommended" and preselected.
/// "Download" starts the usual download (the home screen's DownloadPanel shows it); "decide later" keeps the Get Smaug-Decisions card.
struct FirstRunView: View {
    @EnvironmentObject var model: CascadeModel
    let done: () -> Void
    @State private var picked: ModelChoice?
    @State private var wifiOnly = true

    var body: some View {
        ZStack {
            GradientBackground()
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack(spacing: 12) {
                                IconBadge(icon: "box", solid: true, size: 44)
                                Text("Choose your setup").font(.interTitle3(.bold)).lineLimit(1).minimumScaleFactor(0.8)
                                Spacer(minLength: 0)
                            }
                            // word joiners (U+2060) after the hyphens: a model name never breaks across lines
                            Text("Smaug-\u{2060}Audio-\u{2060}Decisions-\u{2060}0.7B is already on this iPhone."
                                 + (model.policy?.mode == .fastOnly ? "" : " Add Smaug-\u{2060}Decisions for text, photos and harder audio."))
                                .font(.interFootnote()).foregroundStyle(.secondary).lineLimit(3).fixedSize(horizontal: false, vertical: true)
                        }
                        if let mode = model.policy?.mode { options(mode) } else { checking }
                    }
                    .padding(.horizontal, 20).padding(.top, 24).padding(.bottom, 24)
                }
                if let mode = model.policy?.mode { actions(mode).padding(.horizontal, 20).padding(.top, 12).padding(.bottom, 4) }
            }
        }
    }

    private func selection(_ mode: DevicePolicy.Mode) -> ModelChoice {
        picked.flatMap { model.offeredChoices.contains($0) ? $0 : nil } ?? FirstRun.recommended(mode: mode)
    }

    /// The selected option's file when it still has to be downloaded.
    private func download(_ mode: DevicePolicy.Mode) -> ModelManifest.Variant? { model.plan(for: selection(mode)).download }

    @ViewBuilder private func options(_ mode: DevicePolicy.Mode) -> some View {
        let sel = selection(mode)
        VStack(spacing: 10) {
            SectionTitle("What to install")
            ForEach(model.offeredChoices) { c in
                Button { withAnimation(.easeInOut(duration: 0.15)) { picked = c } } label: {
                    ChoiceRow(choice: c, selected: c == sel, recommended: c == FirstRun.recommended(mode: mode))
                }
                .buttonStyle(.plain)
            }
            if mode == .fastOnly {
                Text("Not enough memory for Smaug-Decisions on this iPhone")
                    .font(.interCaption()).lineLimit(1).minimumScaleFactor(0.85).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4)
            }
        }
        VStack(spacing: 10) {
            SectionTitle("This iPhone")
            VStack(spacing: 0) {
                InfoRow(icon: "server", title: "Free storage", detail: ModelStorage.freeBytes().map(ModelStorage.gb) ?? "?")
                if sel.usesBig {
                    Divider().padding(.leading, 56)
                    HStack(spacing: 12) {
                        Image(systemName: "wifi").font(.system(size: 15, weight: .medium)).foregroundStyle(.secondary).frame(width: 28)
                        Text("Download on Wi-Fi only").font(.interSubheadline()).lineLimit(1).minimumScaleFactor(0.85)
                        Spacer(minLength: 8)
                        Toggle("Download on Wi-Fi only", isOn: $wifiOnly).labelsHidden().tint(.violet)
                    }
                    .padding(.horizontal, 16).padding(.vertical, 8)
                }
            }
            .card(padding: 0)
        }
    }

    private var checking: some View {
        HStack(spacing: 10) {
            ProgressView().tint(.violet)
            Text("Checking this iPhone…").font(.interFootnote()).foregroundStyle(.secondary).lineLimit(1)
        }
        .card()
    }

    @ViewBuilder private func actions(_ mode: DevicePolicy.Mode) -> some View {
        let sel = selection(mode), v = download(mode)
        let est = v.map { model.manifest?.isEstimated($0) ?? false } ?? false
        VStack(spacing: 2) {
            if let v, let why = model.downloadBlockerShort(v) {   // next to the button it disables, never below the fold
                HStack(spacing: 8) {
                    Lucide("triangle-alert", size: 15).foregroundStyle(.orange)
                    Text(why).font(.interFootnote(.semibold)).lineLimit(1).minimumScaleFactor(0.85)
                }
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 4).padding(.bottom, 8)
            }
            GradientCapsuleButton(title: v.map { "Download " + (est ? "≈ " : "") + ModelStorage.gb2($0.bytes) } ?? "Continue",
                                  icon: v == nil ? nil : "circle-arrow-down", enabled: v.flatMap(model.downloadBlocker) == nil) { confirm(sel) }
            if sel.usesBig {
                Button(action: later) {
                    Text("Start with the 0.7B audio model, decide later").font(.interSubheadline(.semibold)).foregroundStyle(Color.violet)
                        .lineLimit(1).minimumScaleFactor(0.8).frame(maxWidth: .infinity).padding(.vertical, 10)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func confirm(_ c: ModelChoice) {
        UserDefaults.standard.set(true, forKey: FirstRun.seenKey)
        model.apply(c, deleteOthers: false)   // nothing to delete: the screen only shows while no Smaug-Decisions file is here
        if c.usesBig { model.startDownload(wifiOnly: wifiOnly) }
        done()
    }

    private func later() {
        UserDefaults.standard.set(true, forKey: FirstRun.seenKey)
        done()
    }
}

extension FirstRun {
    /// At launch: not seen yet (a choice stored from the Models screen counts as seen), no .litertlm in Documents (downloaded,
    /// copied in, or awaiting a rename) and no download staged.
    static func shouldShowNow(defaults: UserDefaults = .standard) -> Bool {
        let fm = FileManager.default
        let docs = (try? fm.contentsOfDirectory(atPath: URL.documentsDirectory.path)) ?? []
        let staged = (try? fm.contentsOfDirectory(atPath: ModelDownloader.stagingDir.path)) ?? []
        return shouldShow(seen: defaults.bool(forKey: seenKey) || defaults.string(forKey: ModelChoice.storeKey) != nil,
                          bigOnDevice: docs.contains { $0.hasSuffix(".litertlm") }, downloadPending: staged.contains { $0.hasSuffix(".item.json") })
    }
}
