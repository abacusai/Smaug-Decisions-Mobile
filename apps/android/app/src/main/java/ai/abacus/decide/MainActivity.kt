package ai.abacus.decide

import android.Manifest
import android.content.ComponentCallbacks2
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject
import kotlin.concurrent.thread

/** One answered (or flagged) question, for the result cards. */
class Outcome(val n: Int, val title: String, val kind: String, val r: Cascade.Result, val tau: Double, val memMb: Double)

/** The screen's state: models, download, prepare, results. Engines run on worker threads; state is Compose snapshot state. */
class CascadeUiState(private val act: MainActivity) : AndroidCascade.PrepareListener {
    val manager = ModelManager(act)
    var cascade: Cascade? = null
    var scenarios: List<Scenario> = emptyList()
    var fastReady by mutableStateOf(false)
    var fastState by mutableStateOf("loading")
    var busy by mutableStateOf(false)
    var status by mutableStateOf("")
    var tau by mutableDoubleStateOf(Cascade.DEFAULT_TAU)
    var choice by mutableStateOf(manager.choice)
    var download by mutableStateOf(DownloadUi("idle"))
    var preparing by mutableStateOf<Pair<Long, Double>?>(null)
    var bigLoaded by mutableStateOf(false)
    var tick by mutableIntStateOf(0)   // bumps after file changes so derived values recompute
    val outcomes = mutableStateListOf<Outcome>()
    var developerLog by mutableStateOf("")
    /** the first-run "Choose your setup" screen is up (FirstRun) */
    var firstRun by mutableStateOf(manager.firstRunPending())

    fun load() = thread {
        val t0 = System.nanoTime()
        try {
            runCatching { manager.migrateRenamed() }.getOrNull()?.takeIf { it.isNotEmpty() }?.let { act.runOnUiThread { tick++ } }
            val f = AndroidCascade.fastTier(act); scenarios = AndroidCascade.scenarios(act)
            cascade = AndroidCascade.cascade(act, f, tier = manager.tier, manager = manager, listener = this)
            act.runOnUiThread { fastReady = true; fastState = "ready (%.1f s)".format((System.nanoTime() - t0) / 1e9) }
        } catch (e: Throwable) { act.runOnUiThread { fastState = "failed: ${e.message}" } }
    }

    override fun preparing(startMs: Long, expectedS: Double) { act.runOnUiThread { preparing = startMs to expectedS } }
    override fun prepared(seconds: Double) { act.runOnUiThread { preparing = null; status = "" } }

    fun setChoice(c: ModelChoice, deleteOthers: Boolean) {
        manager.apply(c, deleteOthers)?.let { status = it }
        if (cascade?.bigLoaded == true) cascade?.releaseBig()
        cascade?.choice = manager.choice; choice = manager.choice; bigLoaded = false; tick++
    }

    /** Leaves the first-run screen: [c] is applied and its download started, or null to decide later (the Get Smaug-Decisions card stays). */
    fun finishFirstRun(c: ModelChoice?, wifiOnly: Boolean) {
        manager.markFirstRunSeen()
        if (c != null) {
            setChoice(c, false)   // nothing to delete: the screen only shows while no Smaug-Decisions file is here
            if (c.usesBig) manager.startDownload(wifiOnly)?.let { status = it }
        }
        firstRun = false
    }

    fun onDownload(d: DownloadUi) {
        download = d
        manager.downloadPercent = if (d.active) d.percent else -1
        if (d.phase == "done") { ModelStorage.dropStaleCaches(manager.cacheDir, manager.installedFiles, manager.dir); status = "Smaug-Decisions downloaded and verified."; tick++ }
    }

    fun decide(title: String, kind: String, req: Cascade.Request) {
        val c = cascade ?: return
        busy = true; status = if (kind == "audio") "0.7B audio model…" else "Smaug-Decisions…"
        c.tau = tau
        thread {
            val r = try { c.decide(req) } catch (e: Throwable) { null.also { act.runOnUiThread { status = "error: ${e.message}" } } }
            act.runOnUiThread {
                busy = false; bigLoaded = c.bigLoaded; preparing = null
                if (r != null) { outcomes.add(0, Outcome(outcomes.size + 1, title, kind, r, tau, SelfTest.memoryMb().second)); status = "" }
            }
        }
    }

    fun prepareNow() {
        val c = cascade ?: return
        val s = scenarios.firstOrNull { it.kind == "text" } ?: return
        decide("Prepare Smaug-Decisions", "text", Cascade.Request.Text(s, s.text ?: "test"))
    }

    fun release() { cascade?.releaseBig(); bigLoaded = false; status = "Smaug-Decisions released" }

    val onWifi: Boolean get() {
        val cm = act.getSystemService(ConnectivityManager::class.java)
        return cm.getNetworkCapabilities(cm.activeNetwork)?.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) != true
    }
}

class MainActivity : ComponentActivity() {
    lateinit var ui: CascadeUiState

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        ui = CascadeUiState(this); ui.load()
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 2)
        setContent { CascadeTheme { CascadeScreen(ui) } }
    }

    fun needMic(): Boolean {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) return false
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1); return true
    }

    /** Memory pressure: drop Smaug-Decisions; the next escalation reloads it. */
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        @Suppress("DEPRECATION")
        if (level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND || level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW ||
            level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) { ui.cascade?.releaseBig(); ui.bigLoaded = false }
    }

    override fun onDestroy() { ui.cascade?.close(); super.onDestroy() }
}

// MARK: screen

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CascadeScreen(ui: CascadeUiState) {
    var showModels by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { ui.manager.downloadFlow().collect { ui.onDownload(DownloadUi.of(it)) } }
    if (ui.firstRun) { FirstRunScreen(ui); return }
    LazyColumn(Modifier.fillMaxSize().screenGradient(ui.outcomes.isNotEmpty()).statusBarsPadding().navigationBarsPadding().padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item { Header(ui) { showModels = true } }
        ui.preparing?.let { (start, exp) -> item { PreparingCard(start, exp) } }
        if (ui.tick >= 0 && (ui.manager.pendingDownload != null || ui.download.active)) item { GetModelCard(ui) { showModels = true } }
        item { AskCard(ui) }
        if (ui.status.isNotEmpty()) item { Line(ui.status, Type.caption, secondaryText, Modifier.padding(horizontal = 4.dp), maxLines = 2) }
        if (ui.outcomes.isNotEmpty()) item { SectionTitle("Results") }
        items(ui.outcomes, key = { it.n }) { ResultCard(it, compact = it !== ui.outcomes.first()) }
        item { Spacer(Modifier.height(24.dp)) }
    }
    if (showModels) {
        ModalBottomSheet(onDismissRequest = { showModels = false }, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.background) { ModelsSheet(ui) { showModels = false } }
    }
}

@Composable
fun Header(ui: CascadeUiState, openModels: () -> Unit) {
    Column(Modifier.padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Line("Cascade", Type.title2)
                Line((if (ui.manager.tier == DeviceTier.FAST_ONLY) "Fast only" else ui.choice.title) + " · on this phone", Type.footnote, secondaryText)
            }
            Box(Modifier.size(40.dp).clip(CircleShape).background(cardFill).border(1.dp, hairline, CircleShape).clickable(onClick = openModels),
                contentAlignment = Alignment.Center) { Text("⚙", fontSize = 18.sp) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Pill("0.7B audio model", if (ui.fastReady) "ready" else "loading", if (ui.fastReady) SuccessGreen else Amber)
            val a = bigShort(ui)
            Pill("Smaug-Decisions", a, when (a) { "ready", "loaded" -> SuccessGreen; "off", "not installed" -> secondaryText.copy(alpha = 0.4f); else -> Amber })
        }
    }
}

fun bigShort(ui: CascadeUiState): String = when {
    ui.manager.tier == DeviceTier.FAST_ONLY -> "off"
    !ui.choice.usesBig -> "not installed"
    ui.manager.activeVariant == null -> if (ui.download.active) "${ui.download.percent}%" else "not installed"
    ui.preparing != null -> "preparing"
    ui.bigLoaded -> "loaded"
    else -> "ready"
}

@Composable
fun Pill(name: String, state: String, dot: Color) {
    Row(Modifier.clip(CircleShape).background(cardFill).border(1.dp, hairline, CircleShape).padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(dot))
        // one text, so name and state shrink together: both pills share one row and never wrap or run off the screen
        val sec = secondaryText
        Text(buildAnnotatedString {
            withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) { append(name) }
            withStyle(SpanStyle(color = sec, fontWeight = FontWeight.Normal)) { append("  $state") }
        }, style = Type.caption, maxLines = 1, softWrap = false, overflow = TextOverflow.Ellipsis,
            autoSize = TextAutoSize.StepBased(minFontSize = 8.sp, maxFontSize = 12.sp, stepSize = 0.5.sp))
    }
}

@Composable
fun PreparingCard(start: Long, expected: Double) {
    var now by remember { mutableStateOf(System.currentTimeMillis()) }
    LaunchedEffect(start) { while (true) { delay(500); now = System.currentTimeMillis() } }
    val t = (now - start) / 1000.0
    Card {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Badge("⚙", solid = true, size = 44.dp)
            Column {
                Line("Preparing Smaug-Decisions", Type.title3)
                Line(if (t < expected) "%.0f s of about %.0f s".format(t, expected) else "%.0f s · almost done".format(t), Type.footnote, secondaryText)
            }
        }
        Track((t / expected).toFloat().coerceAtMost(0.97f))
        Line("One-time setup. Later loads are fast.", Type.caption, secondaryText)
    }
}

@Composable
fun GetModelCard(ui: CascadeUiState, openModels: () -> Unit) {
    Card {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Badge("↓", solid = true, size = 44.dp)
            Column {
                Line("Get Smaug-Decisions", Type.title3)
                Line("Text, photos and unsure audio", Type.footnote, secondaryText)
            }
        }
        FieldRow("Use", ui.choice.title, openModels)
        DownloadPanel(ui)
    }
}

@Composable
fun DownloadPanel(ui: CascadeUiState) {
    val d = ui.download; val m = ui.manager
    var wifiOnly by remember { mutableStateOf(false) }
    when (d.phase) {
        "downloading", "queued" -> {
            Row { Line(if (d.phase == "queued") "Waiting for network" else "Downloading", Type.subhead); Spacer1(); Line("${d.percent}%", Type.mono, Violet) }
            Track(d.percent / 100f)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Line("${ModelStorage.gb2(d.received)} of ${ModelStorage.gb2(d.total)}", Type.caption, secondaryText); Spacer1()
                SecondaryButton("Pause") { m.pauseDownload() }
            }
        }
        "verifying" -> { Line("Checking the file (sha256)", Type.subhead); Track(d.verify.toFloat()) }
        "done" -> if (m.pendingDownload == null) Line("✓ Smaug-Decisions downloaded and verified", Type.subhead, SuccessGreen)
        else -> {}
    }
    val v = m.pendingDownload
    if (v != null && !d.active) {
        if (d.phase == "paused" || d.phase == "failed") Line(d.message, Type.footnote, Warn, maxLines = 2)
        val est = m.manifest?.isEstimated(v) == true
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatChip("File", (if (est) "≈" else "") + ModelStorage.gb2(v.bytes))
            StatChip("Needs", ModelStorage.gb(ModelStorage.downloadNeed(v, m.partBytes(v))))
            StatChip("Free", ModelStorage.gb(m.freeBytes()))
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Line(if (ui.onWifi) "Wi-Fi recommended" else "On mobile data: Wi-Fi recommended", Type.caption, if (ui.onWifi) secondaryText else Warn, Modifier.weight(1f))
            Line("Wi-Fi only", Type.caption, secondaryText); Spacer(Modifier.width(6.dp))
            Switch(wifiOnly, { wifiOnly = it }, colors = SwitchDefaults.colors(checkedTrackColor = Violet))
        }
        m.downloadBlocker(v)?.let { Line(it, Type.footnote, Warn, maxLines = 2) }
        val resume = m.partBytes(v) > 0
        GradientButton(if (resume) "Resume" else "Download", enabled = m.downloadBlocker(v) == null) {
            m.startDownload(wifiOnly)?.let { ui.status = it }
        }
    }
}

// MARK: ask

@Composable
fun AskCard(ui: CascadeUiState) {
    var mode by remember { mutableStateOf("Examples") }
    Card {
        Line("Ask", Type.title3)
        // a segmented track: the chips sit inside one capsule with a 3 dp inset, so the last one never touches the card edge
        Row(Modifier.fillMaxWidth().clip(CircleShape).background(fieldFill).padding(3.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            for (m in listOf("Audio", "Text", "Photo", "Examples")) {
                Box(Modifier.weight(1f).clip(CircleShape).background(if (m == mode) Violet else Color.Transparent).clickable { mode = m }.padding(vertical = 8.dp),
                    contentAlignment = Alignment.Center) { Line(m, Type.captionBold, if (m == mode) Color.White else MaterialTheme.colorScheme.onSurface) }
            }
        }
        when (mode) {
            "Audio" -> AudioInput(ui)
            "Text" -> TypedInput(ui, image = false)
            "Photo" -> TypedInput(ui, image = true)
            else -> Examples(ui)
        }
    }
}

class Recording { @Volatile var stop = false; val pcm = FloatArray(16000 * 10); @Volatile var n = 0 }

@Composable
fun AudioInput(ui: CascadeUiState) {
    val act = androidx.compose.ui.platform.LocalContext.current as MainActivity
    val audio = ui.scenarios.filter { it.kind == "audio" }
    var sel by remember { mutableIntStateOf(0) }
    var rec by remember { mutableStateOf<Recording?>(null) }
    var recording by remember { mutableStateOf(false) }
    var seconds by remember { mutableDoubleStateOf(0.0) }
    var clip by remember { mutableStateOf<FloatArray?>(null) }
    LaunchedEffect(recording) { while (recording) { delay(100); seconds = (rec?.n ?: 0) / 16000.0 } }
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        audio.forEachIndexed { i, s -> Chip(s.title, if (i == sel) Violet else secondaryText, Modifier.clickable { sel = i }) }
    }
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
        Box(Modifier.size(56.dp).clip(CircleShape).background(if (recording) Color(0xFFFF3B30) else Violet).clickable(enabled = !ui.busy) {
            if (recording) { rec?.stop = true; recording = false }
            else if (!act.needMic()) {
                val r = Recording(); rec = r; recording = true; seconds = 0.0
                thread {
                    val min = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
                    @Suppress("MissingPermission")
                    val ar = AudioRecord(MediaRecorder.AudioSource.MIC, 16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, 64000))
                    ar.startRecording()
                    while (!r.stop && r.n < r.pcm.size) { val k = ar.read(r.pcm, r.n, minOf(1600, r.pcm.size - r.n), AudioRecord.READ_BLOCKING); if (k <= 0) break; r.n += k }
                    ar.stop(); ar.release()
                    act.runOnUiThread { clip = r.pcm.copyOf(r.n); recording = false }
                }
            }
        }, contentAlignment = Alignment.Center) { Text(if (recording) "■" else "●", color = Color.White, fontSize = 22.sp) }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Line(if (recording) "Recording" else if (clip == null) "Tap to record (10 s max)" else "Clip ready", Type.subhead)
            Track((seconds / 10).toFloat(), if (recording) Color(0xFFFF3B30) else Violet)
            Line("%.1f s".format(seconds), Type.mono, secondaryText)
        }
    }
    GradientButton("Decide", enabled = !ui.busy && clip != null && audio.isNotEmpty()) {
        val s = audio[sel]; ui.decide(s.title, "audio", Cascade.Request.Audio(s, clip!!))
    }
}

@Composable
fun Field(value: String, placeholder: String, onChange: (String) -> Unit, lines: Int = 1) {
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(fieldFill).padding(horizontal = 14.dp, vertical = 12.dp)) {
        if (value.isEmpty()) Line(placeholder, Type.body.copy(fontSize = 15.sp), secondaryText)
        BasicTextField(value, onChange, textStyle = Type.body.copy(fontSize = 15.sp, color = MaterialTheme.colorScheme.onSurface), maxLines = lines,
            singleLine = lines == 1, cursorBrush = SolidColor(Violet), modifier = Modifier.fillMaxWidth())
    }
}

@Composable
fun TypedInput(ui: CascadeUiState, image: Boolean) {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    var text by remember { mutableStateOf("") }
    var question by remember { mutableStateOf("") }
    var options by remember { mutableStateOf("Yes, No") }
    var photo by remember { mutableStateOf<Uri?>(null) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { photo = it }
    if (image) FieldRow("Photo", if (photo == null) "Choose a photo" else "Photo chosen") { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    else Field(text, "Text to decide about", { text = it }, lines = 4)
    Field(question, "Question", { question = it }, lines = 2)
    Field(options, "Options, comma-separated", { options = it })
    val opts = options.split(",").map { it.trim() }.filter { it.isNotEmpty() }
    GradientButton("Decide", enabled = !ui.busy && question.isNotEmpty() && opts.size in 2..Prompt.LETTERS.length && (if (image) photo != null else text.isNotEmpty())) {
        val kind = if (image) "image" else "text"
        val s = Scenario(JSONObject().put("id", "typed").put("kind", kind).put("title", question).put("question", question).put("options", JSONArray(opts)), 0)
        val req = if (image) {
            val bytes = ctx.contentResolver.openInputStream(photo!!)!!.use { it.readBytes() }
            val rgb = AndroidSelfTest.decodeBitmap(bytes); Cascade.Request.Image(s, Decider.Input.fromRgb(rgb.rgb, rgb.width, rgb.height))
        } else Cascade.Request.Text(s, text)
        ui.decide(question, kind, req)
    }
}

/** Built-in examples from the cascade self-test bundle (our own clips and texts, no benchmark items). */
val EXAMPLES = listOf("complaint:speech_activity", "unfinished:turn_complete", "quiet_room:intent", "music_only:speech_activity", "text:triage:36", "image:portrait.png")

@Composable
fun Examples(ui: CascadeUiState) {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val reqs = remember(ui.fastReady) {
        runCatching { JSONObject(ctx.assets.open("cascade_selftest/expected.json").use { String(it.readBytes()) }).getJSONArray("requests") }.getOrNull()
            ?.let { a -> (0 until a.length()).map { a.getJSONObject(it) }.filter { it.getString("id") in EXAMPLES } } ?: emptyList()
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        for (q in reqs) {
            val s = ui.scenarios.firstOrNull { it.id == q.getString("scenario") } ?: continue
            val kind = q.getString("kind")
            val clip = if (kind == "text") "text" else q.getString("input").substringBefore('.').replace('_', ' ')
            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(fieldFill).clickable(enabled = !ui.busy && ui.fastReady) {
                val req = when (kind) {
                    "audio" -> Cascade.Request.Audio(s, AudioIn.load(ctx.assets.open("cascade_selftest/${q.getString("input")}").use { it.readBytes() }))
                    "text" -> Cascade.Request.Text(s, q.getString("input"))
                    else -> AndroidSelfTest.decodeBitmap(ctx.assets.open("cascade_selftest/${q.getString("input")}").use { it.readBytes() })
                        .let { Cascade.Request.Image(s, Decider.Input.fromRgb(it.rgb, it.width, it.height)) }
                }
                ui.decide(s.title, kind, req)
            }.padding(10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Badge(when (kind) { "audio" -> "♪"; "text" -> "T"; else -> "▣" })
                Column(Modifier.weight(1f)) { Line(s.title, Type.subhead); Line("$kind · $clip", Type.caption, secondaryText) }
                Line("›", Type.subhead, secondaryText)
            }
        }
    }
}

// MARK: results

fun routeLabel(r: Cascade.Route) = when (r) {
    Cascade.Route.FAST -> "0.7B audio model"; Cascade.Route.ESCALATED -> "Escalated"; Cascade.Route.ROUTED -> "Smaug-Decisions"
    Cascade.Route.FLAGGED -> "Flagged"; Cascade.Route.UNSURE -> "Unsure"; Cascade.Route.NOT_INSTALLED -> "Not installed"
}

fun routeColor(r: Cascade.Route) = when (r) {
    Cascade.Route.FAST -> SuccessGreen; Cascade.Route.ESCALATED -> Warn; Cascade.Route.ROUTED -> Violet
    Cascade.Route.FLAGGED, Cascade.Route.UNSURE -> Color(0xFFAF52DE); Cascade.Route.NOT_INSTALLED -> Color.Gray
}

@Composable
fun ResultCard(o: Outcome, compact: Boolean) {
    var details by remember { mutableStateOf(false) }
    val r = o.r
    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Chip(routeLabel(r.route), routeColor(r.route)); Spacer(Modifier.width(8.dp))
            Line(o.title, Type.caption, secondaryText, Modifier.weight(1f).padding(start = 4.dp))
        }
        Column {
            Line(if (r.answered) r.top.first else r.note, if (compact) Type.subhead else Type.title3)
            if (r.answered) Line("by " + when (r.route) { Cascade.Route.FAST -> "0.7B audio model"; Cascade.Route.FLAGGED, Cascade.Route.UNSURE -> "0.7B audio model (unsure)"; else -> "Smaug-Decisions" },
                Type.caption, secondaryText)
        }
        r.fastConfidence?.let { c ->
            if (!compact) {
                Row { FitLine("0.7B audio model confidence", Type.caption, secondaryText, Modifier.weight(1f))
                    Line("%.2f %s τ %.2f".format(c, if (c >= o.tau) "≥" else "<", o.tau), Type.mono, if (c >= o.tau) SuccessGreen else Warn) }
                Track(c.toFloat(), if (c >= o.tau) SuccessGreen else Warn, marker = o.tau.toFloat())
            }
        }
        if (r.answered && !compact) {
            val top = r.p.indices.sortedByDescending { r.p[it] }.take(5).sorted()
            val best = r.p.indices.maxBy { r.p[it] }
            for (i in top) Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Line(r.options[i], Type.footnote, modifier = Modifier.weight(1f))
                    if (r.bigP != null && r.fastP != null) Line("0.7B audio model %.2f".format(r.fastP[i]), Type.caption, secondaryText, Modifier.padding(end = 8.dp))
                    Line("%.3f".format(r.p[i]), Type.mono)
                }
                Track(r.p[i].toFloat(), if (i == best) Violet else secondaryText.copy(alpha = 0.6f))
            }
        }
        if (r.answered && r.note.isNotEmpty()) Line(r.note, Type.caption, Warn, maxLines = 2)
        Row(Modifier.fillMaxWidth().clickable { details = !details }, verticalAlignment = Alignment.CenterVertically) {
            Line(if (details) "Details ▾" else "Details ›", Type.captionBold, Violet); Spacer1()
            Line("%.2f s".format((r.msFast + r.msBig) / 1000), Type.mono, secondaryText)
        }
        if (details) Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            if (r.fastP != null) StatChip("0.7B audio model", "%.0f ms".format(r.msFast), 168.dp)
            if (r.bigP != null) StatChip("Smaug-Decisions", "%.2f s".format(r.msBig / 1000), 156.dp)
            if (r.bigLoadMs > 0) StatChip("Load", "%.1f s".format(r.bigLoadMs / 1000))
            StatChip("Memory", "%.0f MB".format(o.memMb), 120.dp)
            StatChip("τ", "%.2f".format(o.tau), 72.dp)
        }
        if (details && r.detail.isNotEmpty()) Line(r.detail, Type.caption, secondaryText, maxLines = 6)
    }
}

// MARK: models

@Composable
fun ModelsSheet(ui: CascadeUiState, done: () -> Unit) {
    val m = ui.manager
    var confirm by remember { mutableStateOf<ModelPlan?>(null) }
    Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).padding(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Badge("◫", solid = true, size = 44.dp)
            Column { Line("Models", Type.title3); Line("What runs on this phone", Type.footnote, secondaryText) }
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("What to install")
            for (c in m.offered) {
                val p = m.plan(c)
                ChoiceRow(c, cost(m, c, p), c == ui.choice) { if (p.delete.isEmpty()) ui.setChoice(c, false) else confirm = p }
            }
            if (m.tier == DeviceTier.FAST_ONLY) Line("Not enough memory for Smaug-Decisions on this phone", Type.caption, secondaryText)
        }
        if (ui.tick >= 0 && (m.pendingDownload != null || ui.download.active)) Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Download"); Card { DownloadPanel(ui) }
        }
        if (ui.choice.escalatesAudio) Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Cascade")
            Card {
                Row { Line("Escalate below", Type.subhead); Spacer1(); Line("τ %.2f".format(ui.tau), Type.mono, Violet) }
                Slider(ui.tau.toFloat(), { ui.tau = (Math.round(it * 100) / 100.0) }, valueRange = 0.5f..0.99f,
                    colors = SliderDefaults.colors(thumbColor = Violet, activeTrackColor = Violet))
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("This device")
            Card(padding = 0.dp) {
                val mi = AndroidCascade.memoryInfo(androidx.compose.ui.platform.LocalContext.current)
                InfoLine("Tier", m.tier.name.lowercase().replace('_', ' '))
                InfoLine("RAM", "%.1f GB · %.1f GB free".format(mi.totalMem / 1e9, mi.availMem / 1e9))
                InfoLine("Free storage", ModelStorage.gb(m.freeBytes()))
                InfoLine("0.7B audio model", ui.fastState)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) { InfoLine("Smaug-Decisions", (m.activeVariant?.variant ?: "none") + " · " + bigShort(ui)) }
                    if (m.activeVariant != null && !ui.bigLoaded && !m.isPrepared(m.activeVariant!!)) Box(Modifier.padding(end = 12.dp)) { SecondaryButton("Prepare now", !ui.busy) { ui.prepareNow() } }
                    if (ui.bigLoaded) Box(Modifier.padding(end = 12.dp)) { SecondaryButton("Release", !ui.busy) { ui.release() } }
                }
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Developer")
            Card {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    val ctx = androidx.compose.ui.platform.LocalContext.current
                    SecondaryButton("Cascade self-test", !ui.busy) {
                        val c = ui.cascade ?: return@SecondaryButton
                        ui.busy = true; ui.developerLog = "running…"
                        thread { val s = runCatching { AndroidCascade.selfTest(ctx, c).getJSONObject("summary").toString(1) }.getOrElse { "error: ${it.message}" }
                            (ctx as MainActivity).runOnUiThread { ui.busy = false; ui.developerLog = s } }
                    }
                    SecondaryButton("Smaug-Decisions self-test", !ui.busy) {
                        ui.busy = true; ui.cascade?.releaseBig(); ui.developerLog = "running…"
                        thread { val s = runCatching { AndroidSelfTest.run(ctx).getJSONObject("summary").toString(1) }.getOrElse { "error: ${it.message}" }
                            (ctx as MainActivity).runOnUiThread { ui.busy = false; ui.developerLog = s } }
                    }
                }
                if (ui.developerLog.isNotEmpty()) Text(ui.developerLog, style = Type.caption, color = secondaryText, maxLines = 12)
            }
        }
        GradientButton("Done", onClick = done)
    }
    confirm?.let { p ->
        AlertDialog(onDismissRequest = { confirm = null },
            title = { Line("Delete files you no longer need?", Type.subhead) },
            text = { Text(p.delete.joinToString("\n") { "${it.file} (${ModelStorage.gb2(it.bytes)})" } + "\nFrees ${ModelStorage.gb(p.freedBytes)}.", style = Type.footnote) },
            confirmButton = { TextButton({ ui.setChoice(p.choice, true); confirm = null }) { Line("Delete and switch", Type.captionBold, Color(0xFFFF3B30)) } },
            dismissButton = { TextButton({ ui.setChoice(p.choice, false); confirm = null }) { Line("Keep files", Type.captionBold, Violet) } })
    }
}

/** One choice with what it costs on this device (Models screen, first-run screen). */
@Composable
fun ChoiceRow(c: ModelChoice, cost: String, selected: Boolean, recommended: Boolean = false, detail: String = c.detail,
              onClick: () -> Unit) {
    Card(Modifier.clickable(onClick = onClick), selected = selected, padding = 12.dp, radius = 14.dp) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Badge(when (c) { ModelChoice.AUDIO_ONLY -> "♪"; ModelChoice.TEXT_IMAGE -> "▣"; ModelChoice.AUDIO_TEXT -> "≋"; ModelChoice.FULL -> "✦" }, solid = selected)
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FitLine(c.title, Type.subhead, modifier = Modifier.weight(1f, fill = false))
                    if (recommended) Chip("Recommended")
                }
                FitLine(detail, Type.caption, secondaryText)
                Line(cost, Type.caption2, if (selected) Violet else secondaryText)
            }
            Radio(selected)
        }
    }
}

fun cost(m: ModelManager, c: ModelChoice, p: ModelPlan): String {
    if (!c.usesBig) return "Nothing to download"
    val v = p.target ?: return "Not in the manifest"
    val est = m.manifest?.isEstimated(v) == true
    val size = (if (est) "≈" else "") + ModelStorage.gb2(v.bytes)
    val disk = ModelStorage.gb(v.bytes + v.cacheBytes)
    return if (p.download == null) "Installed · $size · $disk with cache" else "Download $size · $disk with cache"
}
