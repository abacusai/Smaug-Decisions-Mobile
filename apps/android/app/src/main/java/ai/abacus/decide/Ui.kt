package ai.abacus.decide

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// The ChatLLM Android app's look (chatllm-android libs/Constants.kt, GradientBackground.kt; the iOS app's design guide):
// violet accent 0xFF8B54FF, one success green, neutral gray surfaces (no Material lavender), Inter type, cards with a hairline
// border on a surface + violet wash background, gradient capsule CTAs. Text never wraps: one line with an ellipsis.

val Violet = Color(0xFF8B54FF)
val VioletEnd = Color(0xFFB65CF0)
val SuccessGreen = Color(0xFF34C759)
val Warn = Color(0xFFFF9500)
/** loading / preparing / downloading (iOS systemOrange) */
val Amber = Warn

val Inter = FontFamily(Font(R.font.inter_regular, FontWeight.Normal), Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold), Font(R.font.inter_bold, FontWeight.Bold))

private val LightColors = lightColorScheme(primary = Violet, secondary = Color(0xFF03DAC6), background = Color(0xFFF2F2F7), surface = Color(0xFFFFFFFF),
    onPrimary = Color.White, onBackground = Color.Black, onSurface = Color.Black, surfaceVariant = Color(0xFFE5E5EA), onSurfaceVariant = Color(0xFF48484A),
    surfaceContainerLowest = Color(0xFFFFFFFF), surfaceContainerLow = Color(0xFFF7F7F8), surfaceContainer = Color(0xFFF2F2F7),
    surfaceContainerHigh = Color(0xFFECECF0), surfaceContainerHighest = Color(0xFFE5E5EA))
private val DarkColors = darkColorScheme(primary = Violet, secondary = Color(0xFF03DAC6), background = Color.Black, surface = Color.Black,
    onPrimary = Color.White, onBackground = Color.White, onSurface = Color.White, surfaceVariant = Color(0xFF2C2C2E), onSurfaceVariant = Color(0xFFC7C7CC),
    surfaceContainerLowest = Color(0xFF0E0E10), surfaceContainerLow = Color(0xFF1C1C1E), surfaceContainer = Color(0xFF1F1F21),
    surfaceContainerHigh = Color(0xFF2A2A2C), surfaceContainerHighest = Color(0xFF353537))

@Composable
fun CascadeTheme(dark: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (dark) DarkColors else LightColors, content = content)
}

object Type {
    val title2 = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Bold, fontSize = 22.sp)
    val title3 = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Bold, fontSize = 19.sp)
    val body = TextStyle(fontFamily = Inter, fontSize = 16.sp)
    val subhead = TextStyle(fontFamily = Inter, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
    val footnote = TextStyle(fontFamily = Inter, fontSize = 13.sp)
    val caption = TextStyle(fontFamily = Inter, fontSize = 12.sp)
    val captionBold = TextStyle(fontFamily = Inter, fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
    val caption2 = TextStyle(fontFamily = Inter, fontWeight = FontWeight.SemiBold, fontSize = 11.sp)
    val mono = TextStyle(fontFamily = Inter, fontWeight = FontWeight.SemiBold, fontSize = 12.sp, fontFeatureSettings = "tnum")
}

val dark: Boolean @Composable get() = MaterialTheme.colorScheme.background == Color.Black
val secondaryText: Color @Composable get() = MaterialTheme.colorScheme.onSurfaceVariant
val cardFill: Color @Composable get() = if (dark) Color.White.copy(alpha = 0.06f) else Color.White
val fieldFill: Color @Composable get() = MaterialTheme.colorScheme.onSurface.copy(alpha = if (dark) 0.07f else 0.04f)
val hairline: Color @Composable get() = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.07f)

/** One line, ellipsis: every title, row, chip, stat and button label. */
@Composable
fun Line(text: String, style: TextStyle, color: Color = Color.Unspecified, modifier: Modifier = Modifier, maxLines: Int = 1) =
    Text(text, style = style, color = color, maxLines = maxLines, overflow = TextOverflow.Ellipsis, modifier = modifier, softWrap = maxLines > 1)

/** One line that shrinks its text (down to 75%) before it falls back to an ellipsis: tight rows that carry a model name. */
@Composable
fun FitLine(text: String, style: TextStyle, color: Color = Color.Unspecified, modifier: Modifier = Modifier) =
    Text(text, style = style, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = modifier, softWrap = false,
        autoSize = TextAutoSize.StepBased(minFontSize = style.fontSize * 0.75f, maxFontSize = style.fontSize, stepSize = 0.5.sp))

/** surface + a violet wash toward the bottom (flipped: toward the top once results exist) */
@Composable
fun Modifier.screenGradient(flipped: Boolean): Modifier {
    val bg = MaterialTheme.colorScheme.background; val v = Violet.copy(alpha = 0.22f)
    return drawBehind {
        drawRect(bg)
        val stops = if (flipped) arrayOf(0f to v, 0.2f to Color.Transparent, 1f to Color.Transparent)
                    else arrayOf(0f to Color.Transparent, 0.25f to Color.Transparent, 1f to v)
        drawRect(Brush.verticalGradient(*stops))
    }
}

@Composable
fun Card(modifier: Modifier = Modifier, selected: Boolean = false, padding: Dp = 16.dp, radius: Dp = 16.dp, content: @Composable ColumnScope.() -> Unit) {
    val shape = RoundedCornerShape(radius)
    Column(modifier.fillMaxWidth().clip(shape)
        .background(if (selected) Violet.copy(alpha = if (dark) 0.18f else 0.08f) else cardFill)
        .border(if (selected) 1.5.dp else 1.dp, if (selected) Violet else hairline, shape).padding(padding),
        verticalArrangement = Arrangement.spacedBy(12.dp), content = content)
}

@Composable
fun SectionTitle(text: String) = Line(text.uppercase(), Type.caption2.copy(letterSpacing = 0.6.sp), secondaryText, Modifier.padding(start = 4.dp))

/** rounded-square badge with a glyph: violet tint, or solid violet for primary / selected */
@Composable
fun Badge(glyph: String, solid: Boolean = false, size: Dp = 36.dp, tint: Color = Violet) {
    Box(Modifier.size(size).clip(RoundedCornerShape(size / 4)).background(if (solid) tint else tint.copy(alpha = if (dark) 0.2f else 0.1f)),
        contentAlignment = Alignment.Center) {
        Text(glyph, color = if (solid) Color.White else tint, fontSize = (size.value * 0.42f).sp, fontFamily = Inter, fontWeight = FontWeight.SemiBold, maxLines = 1)
    }
}

@Composable
fun GradientButton(title: String, enabled: Boolean = true, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Box(modifier.fillMaxWidth().height(50.dp).clip(CircleShape)
        .background(Brush.horizontalGradient(listOf(Violet, VioletEnd))).alpha(if (enabled) 1f else 0.5f)
        .clickable(enabled = enabled, onClick = onClick), contentAlignment = Alignment.Center) {
        Line(title, Type.subhead.copy(fontSize = 16.sp), Color.White)
    }
}

@Composable
fun SecondaryButton(title: String, enabled: Boolean = true, onClick: () -> Unit) {
    Box(Modifier.clip(CircleShape).background(cardFill).border(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f), CircleShape)
        .clickable(enabled = enabled, onClick = onClick).padding(horizontal = 14.dp, vertical = 7.dp).alpha(if (enabled) 1f else 0.5f)) {
        Line(title, Type.captionBold)
    }
}

/** pill: color 12% fill + 30% stroke, one line */
@Composable
fun Chip(text: String, color: Color = Violet, modifier: Modifier = Modifier) {
    Box(modifier.clip(CircleShape).background(color.copy(alpha = 0.12f)).border(1.dp, color.copy(alpha = 0.3f), CircleShape)
        .padding(horizontal = 10.dp, vertical = 4.dp)) { Line(text, Type.caption2, color) }
}

/** a compact fixed-width number chip (tabular digits, never wraps, shrinks long values) */
@Composable
fun StatChip(label: String, value: String, width: Dp = 104.dp) {
    Row(Modifier.width(width).clip(RoundedCornerShape(10.dp)).background(fieldFill).padding(horizontal = 8.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically) {
        FitLine(label, Type.caption, secondaryText, Modifier.weight(1f))
        Line(value, if (value.length > 8) Type.mono.copy(fontSize = 10.sp) else Type.mono)
    }
}

@Composable
fun Radio(selected: Boolean) {
    Box(Modifier.size(22.dp).clip(CircleShape).background(if (selected) Violet else Color.Transparent)
        .border(1.5.dp, if (selected) Violet else secondaryText.copy(alpha = 0.4f), CircleShape), contentAlignment = Alignment.Center) {
        if (selected) Text("✓", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun Track(value: Float, color: Color = Violet, height: Dp = 6.dp, marker: Float? = null) {
    Box(Modifier.fillMaxWidth().height(maxOf(height, if (marker != null) 12.dp else height)), contentAlignment = Alignment.CenterStart) {
        Box(Modifier.fillMaxWidth().height(height).clip(CircleShape).background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)))
        Box(Modifier.fillMaxWidth(value.coerceIn(0.02f, 1f)).height(height).clip(CircleShape)
            .background(if (color == Violet) Brush.horizontalGradient(listOf(Violet, VioletEnd)) else Brush.horizontalGradient(listOf(color, color))))
        if (marker != null) Box(Modifier.fillMaxWidth(marker.coerceIn(0f, 1f)), contentAlignment = Alignment.CenterEnd) {
            Box(Modifier.width(2.dp).height(12.dp).background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)))
        }
    }
}

@Composable
fun FieldRow(label: String, value: String, onClick: (() -> Unit)? = null) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(fieldFill).let { if (onClick != null) it.clickable(onClick = onClick) else it }
        .padding(horizontal = 14.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Line(label, Type.body.copy(fontSize = 15.sp), secondaryText)
        Line(value, Type.subhead, modifier = Modifier.weight(1f).padding(start = 12.dp), color = MaterialTheme.colorScheme.onSurface)
        if (onClick != null) Line(" ›", Type.subhead, secondaryText)
    }
}

@Composable
fun RowScope.Spacer1() = androidx.compose.foundation.layout.Spacer(Modifier.weight(1f))

@Composable
fun InfoLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Line(label, Type.body.copy(fontSize = 15.sp))
        androidx.compose.foundation.layout.Spacer(Modifier.width(12.dp))
        Line(value, Type.footnote.copy(textAlign = androidx.compose.ui.text.style.TextAlign.End), secondaryText, Modifier.weight(1f))
    }
}

