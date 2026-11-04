package ai.abacus.decide

import kotlin.math.max

/**
 * The on-device cascade (docs/ios.md): audio questions go to the fast tier first and are answered there when its confidence
 * is at least [tau]; otherwise they escalate to Smaug-Decisions, which is loaded lazily on the first escalation and released under
 * memory pressure ([releaseBig], called from onTrimMemory). Text and image questions skip the fast tier (it is near chance on
 * text) and are routed straight to Smaug-Decisions. On a [DeviceTier.FAST_ONLY] device (or when Smaug-Decisions cannot be loaded right now) an
 * unsure audio decision keeps the fast answer and is flagged "would escalate".
 *
 * Pure Kotlin apart from the two engines, so the routing runs on the desktop JVM against the real models.
 */
class Cascade(private val fast: FastTier?, private val bigFactory: () -> BigModel, @Volatile var tau: Double = DEFAULT_TAU,
              val deviceTier: DeviceTier = DeviceTier.FULL,
              /** why Smaug-Decisions cannot take a decision of this kind right now (memory, file missing or downloading, storage), or null */
              private val bigBlocker: (kind: String, bigLoaded: Boolean) -> String? = { _, _ -> null }) : AutoCloseable {
    /** Models screen: what Smaug-Decisions is installed for. A kind it does not cover is never sent to it. */
    @Volatile var choice: ModelChoice = ModelChoice.FULL
    @Volatile private var big: BigModel? = null
    var bigLoads = 0; private set
    var lastBigLoadMs = 0.0; private set

    /** FLAGGED: unsure audio kept (would escalate) or text / image unanswered because Smaug-Decisions cannot run now; UNSURE: unsure
     *  audio when the chosen models do not escalate audio; NOT_INSTALLED: a kind the chosen models do not include. */
    enum class Route { FAST, ESCALATED, ROUTED, FLAGGED, UNSURE, NOT_INSTALLED }

    sealed class Request {
        class Audio(val scenario: Scenario, val pcm16k: FloatArray) : Request()
        class Text(val scenario: Scenario, val text: String) : Request()
        class Image(val scenario: Scenario, val image: Decider.Input.Image) : Request()
    }

    class Result(val options: List<String>, val p: DoubleArray, val route: Route, val fastConfidence: Double?, val fastP: DoubleArray?,
                 val bigP: DoubleArray?, val msFast: Double, val msBig: Double, val bigLoadMs: Double, val note: String = "",
                 /** raw runtime text behind a short note (Details only) */
                 val detail: String = "") {
        val tier: String get() = when (route) { Route.FAST, Route.FLAGGED, Route.UNSURE -> "fast"; Route.NOT_INSTALLED -> "none"; else -> "big" }
        val top: Pair<String, Double> get() = p.indices.maxByOrNull { p[it] }!!.let { options[it] to p[it] }
        val answered: Boolean get() = p.isNotEmpty() && !p[0].isNaN()
    }

    val bigLoaded: Boolean get() = big != null

    fun decide(req: Request): Result = when (req) {
        is Request.Audio -> audio(req)
        is Request.Text -> routed(req.scenario, Decider.Input.Text(req.text))
        is Request.Image -> routed(req.scenario, req.image)
    }

    private fun audio(req: Request.Audio): Result {
        val s = req.scenario
        val f = requireNotNull(fast) { "fast tier not loaded" }.run(s, req.pcm16k)
        val c = confidence(f.p, s.multi)
        when (route("audio", c, tau, true, choice)) {
            Route.FAST -> return Result(s.options, f.p, Route.FAST, c, f.p, null, f.msTotal, 0.0, 0.0)
            Route.UNSURE -> return Result(s.options, f.p, Route.UNSURE, c, f.p, null, f.msTotal, 0.0, 0.0, note = "Smaug-Decisions is not installed for audio (Models)")
            else -> {}
        }
        val blocked = if (deviceTier == DeviceTier.FAST_ONLY) "would escalate (fast-only device)" else bigBlocker("audio", bigLoaded)
        if (blocked != null) return Result(s.options, f.p, Route.FLAGGED, c, f.p, null, f.msTotal, 0.0, 0.0, note = blocked)
        val (b, loadMs) = try { bigEngine() } catch (e: Exception) {
            return Result(s.options, f.p, Route.FLAGGED, c, f.p, null, f.msTotal, 0.0, 0.0, note = LOAD_FAILED, detail = e.message ?: e.toString())
        }
        val r = b.evaluate(Decider.Input.Audio(req.pcm16k), s.question, s.options, s.descriptions, s.multi, s.rules)
        return Result(s.options, r.p, Route.ESCALATED, c, f.p, r.p, f.msTotal, r.ms.toDouble(), loadMs)
    }

    private fun routed(s: Scenario, input: Decider.Input): Result {
        val kind = if (input is Decider.Input.Image) "image" else "text"
        val none = DoubleArray(s.options.size) { Double.NaN }
        if (route(kind, null, tau, true, choice) == Route.NOT_INSTALLED)
            return Result(s.options, none, Route.NOT_INSTALLED, null, null, null, 0.0, 0.0, 0.0,
                note = if (!choice.usesBig) "Smaug-Decisions not installed (0.7B audio model only)" else if (kind == "image") "Images not installed" else "Text not installed")
        val blocked = if (deviceTier == DeviceTier.FAST_ONLY) "needs Smaug-Decisions (text and image questions are not answered by the fast tier)" else bigBlocker(kind, bigLoaded)
        if (blocked != null) return Result(s.options, none, Route.FLAGGED, null, null, null, 0.0, 0.0, 0.0, note = blocked)
        val (b, loadMs) = try { bigEngine() } catch (e: Exception) {
            return Result(s.options, none, Route.FLAGGED, null, null, null, 0.0, 0.0, 0.0, note = LOAD_FAILED, detail = e.message ?: e.toString())
        }
        val r = b.evaluate(input, s.question, s.options, s.descriptions, s.multi, s.rules)
        return Result(s.options, r.p, Route.ROUTED, null, null, r.p, 0.0, r.ms.toDouble(), loadMs)
    }

    @Synchronized private fun bigEngine(): Pair<BigModel, Double> {
        big?.let { return it to 0.0 }
        val t = System.nanoTime()
        val d = bigFactory()
        lastBigLoadMs = (System.nanoTime() - t) / 1e6; bigLoads++
        big = d
        return d to lastBigLoadMs
    }

    /** Frees Smaug-Decisions (about 6 GB); the next escalation loads it again. */
    @Synchronized fun releaseBig() { big?.close(); big = null }

    override fun close() { releaseBig(); fast?.close() }

    companion object {
        const val DEFAULT_TAU = 0.9
        const val LOAD_FAILED = "Smaug-Decisions could not load"

        /** The routing rule, the same as the iOS Router.route: text / image go to Smaug-Decisions when the choice covers them (else
         *  NOT_INSTALLED); audio stays on the fast tier at confidence >= tau, unsure audio escalates only when Smaug-Decisions has audio
         *  (else UNSURE). Without Smaug-Decisions available now, FLAGGED. [kind] is "audio", "text" or "image". */
        fun route(kind: String, confidence: Double?, tau: Double, bigAvailable: Boolean, choice: ModelChoice = ModelChoice.FULL): Route {
            if (kind != "audio") {
                if (!choice.answers(kind)) return Route.NOT_INSTALLED
                return if (bigAvailable) Route.ROUTED else Route.FLAGGED
            }
            if (confidence != null && confidence >= tau) return Route.FAST
            if (!choice.escalatesAudio) return Route.UNSURE
            return if (bigAvailable) Route.ESCALATED else Route.FLAGGED
        }

        /** smaug_decide.cascade.confidence, exactly: one answer -> max(p) / max(sum(p), 1e-9); select-all -> min over options of
         *  max(p, 1 - p) (the least certain option). */
        fun confidence(p: DoubleArray, multi: Boolean): Double =
            if (multi) p.minOf { max(it, 1.0 - it) } else p.max() / max(p.sum(), 1e-9)
    }
}

/** What runs locally, from the device's total RAM (ActivityManager.MemoryInfo.totalMem). See [DevicePolicy]. */
enum class DeviceTier { FULL, STANDARD, SQUEEZE, FAST_ONLY }

/**
 * Device tiers for the LiteRT-LM backend (smaug-decisions-full-vis560.litertlm), from memory measured on x86 (docs/devices.md):
 * - fast tier: ~0.65 GB dirty loaded, up to ~1.1 GB after 28 s clips (ORT arena + Java heap).
 * - Smaug-Decisions, engine without the vision encoder: +0.8 GB dirty for a text or audio decision, plus a 2.4-2.65 GB clean file-backed
 *   working set (the XNNPack packed-decoder cache, read for every token).
 * - Smaug-Decisions image decision: the engine is recreated with the vision encoder: about +2.0-2.6 GB dirty while it runs; dropping the
 *   encoder afterwards (and purging the heap) returns to ~0.86 GB.
 * Android's kernel (kswapd) drops clean file-backed pages and swaps anonymous ones to zram; lmkd kills by oom_score_adj (the
 * foreground app is the last candidate) when PSI memory pressure persists. So the dirty footprint must fit next to the OS, and the
 * clean working set should stay in page cache or every prefill re-reads it from flash.
 * totalMem sits below the marketed size (8 GB -> ~7.2-7.7 GiB, 6 GB -> ~5.5 GiB, 4 GB -> ~3.6 GiB):
 * - FULL >= 9.5 GiB (12 GB+): cascade; the vision encoder stays loaded after the first image.
 * - STANDARD 6.5-9.5 GiB (8 GB): cascade; the vision encoder is dropped after each image. Smaug-Decisions needs availMem to cover what
 *   the decision adds on top of what is already loaded: [AVAIL_TEXT_AUDIO_MB] to load it for text / audio, [AVAIL_IMAGE_LOADED_MB] /
 *   [AVAIL_IMAGE_COLD_MB] for an image with it loaded / not loaded.
 * - SQUEEZE 4.5-6.5 GiB (6 GB): the same, with [SQUEEZE_EXTRA_MB] more for images.
 * availMem is a floor, not a ceiling: it leaves out cached background apps, which lmkd kills for the foreground app. The first
 * gates (2.0 GB text / audio, 3.5 GB image on top of a loaded engine) refused every image on an 8 GB phone (2026-11-04).
 * - FAST_ONLY < 4.5 GiB (4 GB): fast tier only (the ~3.3 GB Smaug-Decisions working set does not fit next to the OS).
 */
object DevicePolicy {
    const val FULL_MIN_GIB = 9.5
    const val STANDARD_MIN_GIB = 6.5
    const val SQUEEZE_MIN_GIB = 4.5
    const val FAST_DIRTY_MB = 1100L          // fast tier, worst case measured (JVM test process)
    const val BIG_DIRTY_TEXT_AUDIO_MB = 830L // Smaug-Decisions LiteRT, text 791 / audio 827 MB (JVM, engine without vision)
    const val BIG_DIRTY_IMAGE_MB = 2650L     // Smaug-Decisions LiteRT with the vision encoder after images (JVM, without heap purge)
    const val BIG_CLEAN_WORKING_SET_MB = 2650L
    /** free memory needed to load Smaug-Decisions for a text / audio decision: its dirty memory + headroom (the clean working set is
     *  file-backed, the kernel can drop and re-read it) */
    const val AVAIL_TEXT_AUDIO_MB = 1200L
    /** free memory needed for an image decision with Smaug-Decisions loaded: the vision encoder's extra dirty memory (~1.8 GB) less what
     *  lmkd frees from cached apps */
    const val AVAIL_IMAGE_LOADED_MB = 1500L
    /** the same with Smaug-Decisions not loaded yet */
    const val AVAIL_IMAGE_COLD_MB = 2300L
    const val SQUEEZE_EXTRA_MB = 500L

    fun tier(totalMemBytes: Long): DeviceTier {
        val gib = totalMemBytes / (1024.0 * 1024 * 1024)
        return when {
            gib >= FULL_MIN_GIB -> DeviceTier.FULL
            gib >= STANDARD_MIN_GIB -> DeviceTier.STANDARD
            gib >= SQUEEZE_MIN_GIB -> DeviceTier.SQUEEZE
            else -> DeviceTier.FAST_ONLY
        }
    }

    /** whether Smaug-Decisions may run a decision of this kind ("audio", "text" or "image") now */
    fun canUseBig(tier: DeviceTier, kind: String, availMemBytes: Long, bigLoaded: Boolean): Boolean {
        val availMb = availMemBytes / (1024 * 1024)
        return when (tier) {
            DeviceTier.FULL -> true
            DeviceTier.FAST_ONLY -> false
            else -> if (kind == "image") availMb >= (if (bigLoaded) AVAIL_IMAGE_LOADED_MB else AVAIL_IMAGE_COLD_MB) + (if (tier == DeviceTier.SQUEEZE) SQUEEZE_EXTRA_MB else 0)
                    else bigLoaded || availMb >= AVAIL_TEXT_AUDIO_MB
        }
    }

    /** keep the vision encoder after an image (FULL) or drop it again (smaller devices) */
    fun keepVision(tier: DeviceTier): Boolean = tier == DeviceTier.FULL
}
