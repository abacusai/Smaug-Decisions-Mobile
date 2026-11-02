package ai.abacus.decide

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** One option on the first-run screen, read from the [ModelManager] (the screen itself renders from plain data). */
data class SetupOption(val choice: ModelChoice, val cost: String, val download: ModelManifest.Variant?, val estimated: Boolean, val blocker: String?)

fun setupOptions(m: ModelManager): List<SetupOption> = m.offered.map { c ->
    val p = m.plan(c); val d = p.download
    SetupOption(c, cost(m, c, p), d, d != null && m.manifest?.isEstimated(d) == true, d?.let(m::downloadBlocker))
}

/** First launch: "Choose your setup", before the home screen and before anything downloads (shown while [CascadeUiState.firstRun]). */
@Composable
fun FirstRunScreen(ui: CascadeUiState) {
    val m = ui.manager
    val options = remember { setupOptions(m) }
    FirstRunContent(options, FirstRun.recommended(m.tier), m.tier == DeviceTier.FAST_ONLY, m.freeBytes(),
        onDownload = { c, wifiOnly -> ui.finishFirstRun(c, wifiOnly) }, onLater = { ui.finishFirstRun(null, false) })
}

/** The rows are the Models screen's [ChoiceRow]s, the device's default marked "Recommended" and preselected. */
@Composable
fun FirstRunContent(options: List<SetupOption>, recommended: ModelChoice, fastOnly: Boolean, freeBytes: Long,
                    onDownload: (ModelChoice, Boolean) -> Unit, onLater: () -> Unit) {
    var picked by remember { mutableStateOf(recommended) }
    var wifiOnly by remember { mutableStateOf(true) }
    val sel = options.firstOrNull { it.choice == picked } ?: options.first()
    // not inside a Surface: text without a color takes the theme's, in dark mode too
    CompositionLocalProvider(LocalContentColor provides MaterialTheme.colorScheme.onBackground) {
        Column(Modifier.fillMaxSize().screenGradient(false).statusBarsPadding().navigationBarsPadding().padding(horizontal = 16.dp)) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(top = 24.dp, bottom = 16.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Badge("◫", solid = true, size = 44.dp)
                        Line("Choose your setup", Type.title3)
                    }
                    // word joiners (U+2060) after the hyphens: a model name never breaks across lines
                    Line("Smaug-\u2060Audio-\u2060Decisions-\u20600.7B is already on this phone." +
                        if (fastOnly) "" else " Add Smaug-\u2060Decisions for text, photos and harder audio.", Type.footnote, secondaryText, maxLines = 3)
                }
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("What to install")
                    for (o in options) {
                        ChoiceRow(o.choice, o.cost, o.choice == sel.choice, recommended = o.choice == recommended, detail = o.choice.detail) { picked = o.choice }
                    }
                    if (fastOnly) Line("Not enough memory for Smaug-Decisions on this phone", Type.caption, secondaryText, Modifier.padding(horizontal = 4.dp))
                }
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("This phone")
                    Card(padding = 0.dp) {
                        InfoLine("Free storage", ModelStorage.gb(freeBytes))
                        if (sel.choice.usesBig) Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 12.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Line("Download on Wi-Fi only", Type.body.copy(fontSize = 15.sp), modifier = Modifier.weight(1f))
                            Switch(wifiOnly, { wifiOnly = it }, colors = SwitchDefaults.colors(checkedTrackColor = Violet))
                        }
                    }
                }
            }
            Column(Modifier.padding(top = 8.dp, bottom = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                sel.blocker?.let { Line(it, Type.footnote, Warn, Modifier.padding(horizontal = 4.dp), maxLines = 2) }
                val d = sel.download
                GradientButton(if (d != null) "Download " + (if (sel.estimated) "≈" else "") + ModelStorage.gb2(d.bytes) else "Continue", enabled = sel.blocker == null) {
                    onDownload(sel.choice, wifiOnly)
                }
                if (sel.choice.usesBig) Box(Modifier.fillMaxWidth().clip(CircleShape).clickable(onClick = onLater).padding(vertical = 12.dp), contentAlignment = Alignment.Center) {
                    Line("Start with the 0.7B audio model, decide later", Type.subhead, Violet)
                }
            }
        }
    }
}
