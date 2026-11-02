import SwiftUI

// The ChatLLM iOS app's design system (chatllm-ios app/CLAUDE.md): Inter type, violet accent (#8B54FF, asset color), the
// GradientBackground wash, adaptive cards (radius 14 / 16, continuous, hairline border), gradient capsule CTAs, Lucide icons as
// templates, uppercased tracked section titles. Every surface adapts to dark mode with opacities over semantic colors.

extension Color {
    static let violet = Color("violet")
    static let violetEnd = Color(hex: "B65CF0")

    init(hex: String) {
        let v = UInt64(hex, radix: 16) ?? 0
        self.init(red: Double((v >> 16) & 0xff) / 255, green: Double((v >> 8) & 0xff) / 255, blue: Double(v & 0xff) / 255)
    }
}

extension Font {
    enum InterWeight: String {
        case regular = "Inter18pt-Regular", medium = "Inter18pt-Medium", semibold = "Inter18pt-SemiBold", bold = "Inter18pt-Bold"
    }

    static func inter(_ size: CGFloat, _ weight: InterWeight = .regular, relativeTo style: Font.TextStyle = .body) -> Font {
        .custom(weight.rawValue, size: size, relativeTo: style)
    }
    static func interTitle2(_ w: InterWeight = .bold) -> Font { inter(22, w, relativeTo: .title2) }
    static func interTitle3(_ w: InterWeight = .regular) -> Font { inter(20, w, relativeTo: .title3) }
    static func interBody(_ w: InterWeight = .regular) -> Font { inter(17, w, relativeTo: .body) }
    static func interCallout(_ w: InterWeight = .regular) -> Font { inter(16, w, relativeTo: .callout) }
    static func interSubheadline(_ w: InterWeight = .regular) -> Font { inter(15, w, relativeTo: .subheadline) }
    static func interFootnote(_ w: InterWeight = .regular) -> Font { inter(13, w, relativeTo: .footnote) }
    static func interCaption(_ w: InterWeight = .regular) -> Font { inter(12, w, relativeTo: .caption) }
    static func interCaption2(_ w: InterWeight = .regular) -> Font { inter(11, w, relativeTo: .caption2) }
}

/// systemGroupedBackground + a violet wash toward the bottom (chatllm-ios GradientBackground).
struct GradientBackground: View {
    var isFlipped = false
    var body: some View {
        ZStack {
            Color(.systemGroupedBackground)
            LinearGradient(stops: [.init(color: .clear, location: 0), .init(color: .clear, location: isFlipped ? 0.8 : 0.25),
                                   .init(color: Color.violet.opacity(0.22), location: 1)],
                           startPoint: isFlipped ? .bottom : .top, endPoint: isFlipped ? .top : .bottom)
        }
        .ignoresSafeArea()
    }
}

/// A card on the gradient: white (light) or white 5% (dark), radius 14/16, hairline border.
struct CardModifier: ViewModifier {
    @Environment(\.colorScheme) private var colorScheme
    var radius: CGFloat = 16, padding: CGFloat = 16, selected = false
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: radius, style: .continuous)
                .fill(selected ? Color.violet.opacity(colorScheme == .dark ? 0.18 : 0.08) : (colorScheme == .dark ? Color.white.opacity(0.05) : Color.white)))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous)
                .stroke(selected ? Color.violet : Color.primary.opacity(0.06), lineWidth: selected ? 1.5 : 1))
    }
}

extension View {
    func card(radius: CGFloat = 16, padding: CGFloat = 16, selected: Bool = false) -> some View {
        modifier(CardModifier(radius: radius, padding: padding, selected: selected))
    }
}

/// Uppercased section label: Inter caption semibold, tracking 0.6, secondary.
struct SectionTitle: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text.uppercased()).font(.interCaption(.semibold)).tracking(0.6).foregroundStyle(.secondary).lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading).padding(.leading, 4)
    }
}

/// Lucide icon from the asset catalog, as a template.
struct Lucide: View {
    let name: String, size: CGFloat
    init(_ name: String, size: CGFloat = 18) { self.name = name; self.size = size }
    var body: some View { Image(name).renderingMode(.template).resizable().scaledToFit().frame(width: size, height: size) }
}

/// Rounded-square icon badge: violet-tinted, or solid violet for primary / selected.
struct IconBadge: View {
    @Environment(\.colorScheme) private var colorScheme
    let icon: String
    var solid = false, size: CGFloat = 36, tint: Color = .violet
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size / 4, style: .continuous)
                .fill(solid ? tint : tint.opacity(colorScheme == .dark ? 0.2 : 0.1))
                .frame(width: size, height: size)
            Lucide(icon, size: size * 0.45).foregroundStyle(solid ? Color.white : tint)
        }
    }
}

/// Primary CTA: violet -> pink gradient capsule, white label, optional leading icon. Dimmed when disabled.
struct GradientCapsuleButton: View {
    let title: String
    var icon: String? = nil
    var enabled = true
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let icon { Lucide(icon, size: 16) }
                Text(title).font(.interBody(.semibold)).lineLimit(1).minimumScaleFactor(0.8)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity).padding(.vertical, 14)
            .background(Capsule().fill(LinearGradient(colors: [.violet, .violetEnd], startPoint: .leading, endPoint: .trailing)))
            .opacity(enabled ? 1 : 0.5)
        }
        .buttonStyle(PressableScaleButtonStyle(pressedScale: 0.97))
        .disabled(!enabled)
    }
}

/// Secondary: capsule with the card fill, primary label, hairline border.
struct SecondaryCapsuleButton: View {
    @Environment(\.colorScheme) private var colorScheme
    let title: String
    var icon: String? = nil
    var compact = false
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                if let icon { Lucide(icon, size: 14) }
                Text(title).font(compact ? .interFootnote(.semibold) : .interSubheadline(.semibold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, compact ? 12 : 16).padding(.vertical, compact ? 7 : 11)
            .background(Capsule().fill(colorScheme == .dark ? Color.white.opacity(0.05) : Color.white))
            .overlay(Capsule().stroke(Color.primary.opacity(0.12), lineWidth: 1))
        }
        .buttonStyle(PressableScaleButtonStyle(pressedScale: 0.95))
    }
}

/// Mode chip / pill: color 12% fill + 30% stroke, Inter caption2 medium.
struct Chip: View {
    let text: String
    var icon: String? = nil
    var color: Color = .violet
    var body: some View {
        HStack(spacing: 5) {
            if let icon { Lucide(icon, size: 12) }
            Text(text).font(.interCaption2(.semibold)).lineLimit(1).fixedSize()
        }
        .foregroundStyle(color)
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(Capsule().fill(color.opacity(0.12)))
        .overlay(Capsule().stroke(color.opacity(0.3), lineWidth: 1))
    }
}

/// A compact number chip (latency, memory, size) that never wraps.
struct StatChip: View {
    let text: String
    var icon: String? = nil
    var body: some View {
        HStack(spacing: 4) {
            if let icon { Lucide(icon, size: 11) }
            Text(text).font(.interCaption2(.semibold).monospacedDigit()).lineLimit(1).minimumScaleFactor(0.7)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Capsule().fill(Color.primary.opacity(0.06)))
        .fixedSize()
    }
}

/// Radio indicator: violet circle with a white check when selected (22x22).
struct RadioIndicator: View {
    let selected: Bool
    var body: some View {
        ZStack {
            Circle().fill(selected ? Color.violet : .clear).frame(width: 22, height: 22)
            Circle().stroke(selected ? Color.violet : Color.secondary.opacity(0.4), lineWidth: 1.5).frame(width: 22, height: 22)
            if selected { Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(.white) }
        }
    }
}

/// Springy press (chatllm-ios ChatInputView).
struct PressableScaleButtonStyle: ButtonStyle {
    var pressedScale: CGFloat = 0.9
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.scaleEffect(configuration.isPressed ? pressedScale : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.6), value: configuration.isPressed)
    }
}

/// A thin violet progress track.
struct ProgressTrack: View {
    let value: Double
    var color: Color = .violet
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.08))
                Capsule().fill(LinearGradient(colors: [color, color == .violet ? .violetEnd : color], startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(4, g.size.width * min(max(value, 0), 1)))
            }
        }
        .frame(height: 6)
    }
}
