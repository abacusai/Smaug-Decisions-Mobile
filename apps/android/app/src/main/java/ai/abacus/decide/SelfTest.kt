package ai.abacus.decide

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.abs

/**
 * Replays the reference cases of a self-test bundle (assets/selftest: expected.json + media, written by
 * llama.cpp/make_selftest_bundle.py from the Python path llama_decide.py) through [Decider] and reports, per case, max |dp| vs the
 * Python probabilities, whether the rendered prompt is byte-identical, prompt positions, latency; plus peak memory. Shared by the
 * instrumented device test, the in-app self-test button and the desktop-JVM host e2e test. No Android dependencies: the caller
 * supplies the file reader and the image decoder (BitmapFactory on Android, ImageIO on the desktop).
 */
object SelfTest {
    class Case(val json: JSONObject) {
        val id: String = json.getString("id")
        val kind: String = json.getString("kind")
        val input: String = json.getString("input")
        val question: String = json.getString("question")
        val options: List<String> = json.getJSONArray("options").let { a -> (0 until a.length()).map { a.getString(it) } }
        val descriptions: List<String?>? = json.optJSONArray("descriptions")?.let { a -> (0 until a.length()).map { if (a.isNull(it)) null else a.getString(it) } }
        val multi: Boolean = json.optBoolean("multi", false)
        val rules: String? = if (json.isNull("rules")) null else json.optString("rules")
        val context: String? = if (json.isNull("context")) null else json.optString("context")
        /** rows that only measure the platform's JPEG decoder (expected = the PIL decode); excluded from the parity summary */
        val decoderRow: Boolean = json.optBoolean("decoder_row", false)
        val expectedP: DoubleArray = json.getJSONArray("p").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
        val hfP: DoubleArray? = json.optJSONArray("p_hf")?.let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
    }

    class Bundle(val meta: JSONObject, val cases: List<Case>) {
        val modelFile: String get() = meta.getString("gguf")
        val mmprojFile: String get() = meta.optString("mmproj", "")
        val b: Double get() = meta.getDouble("b")
    }

    fun parse(expectedJson: String): Bundle {
        val o = JSONObject(expectedJson); val a = o.getJSONArray("cases")
        return Bundle(o, (0 until a.length()).map { Case(a.getJSONObject(it)) })
    }

    /** peak resident set (VmHWM) and current RSS of this process in MB, from /proc/self/status (Linux / Android); -1 if unavailable */
    fun memoryMb(): Pair<Double, Double> = try {
        val s = File("/proc/self/status").readLines()
        fun kb(k: String) = s.first { it.startsWith(k) }.split(Regex("\\s+"))[1].toDouble() / 1024.0
        kb("VmHWM:") to kb("VmRSS:")
    } catch (e: Exception) { -1.0 to -1.0 }

    /** /proc/self/smaps_rollup in MB: Rss, Pss, Pss_Anon, Pss_File, Private_Dirty, Anonymous, ... (Linux 4.14+ / Android 10+).
     *  Dirty / anonymous memory must stay resident; Pss_File is clean mmapped model and cache pages the kernel can drop. */
    fun smapsMb(): JSONObject = try {
        val o = JSONObject()
        File("/proc/self/smaps_rollup").readLines().drop(1).forEach { l ->
            val p = l.split(Regex("\\s+")); if (p.size >= 2 && p[0].endsWith(":")) o.put(p[0].trimEnd(':'), p[1].toDouble() / 1024.0)
        }
        o
    } catch (e: Exception) { JSONObject() }

    /** current RSS split from /proc/self/status: anonymous (heap, repacked weights: must stay resident) vs file-backed (mmapped
     *  model files: clean pages the kernel can drop and re-read under pressure), in MB */
    fun rssSplitMb(): JSONObject = try {
        val s = File("/proc/self/status").readLines()
        fun kb(k: String) = s.firstOrNull { it.startsWith(k) }?.split(Regex("\\s+"))?.get(1)?.toDouble()?.div(1024.0) ?: -1.0
        JSONObject().put("rss_mb", kb("VmRSS:")).put("anon_mb", kb("RssAnon:")).put("file_mb", kb("RssFile:")).put("peak_mb", kb("VmHWM:"))
    } catch (e: Exception) { JSONObject() }

    fun input(c: Case, readFile: (String) -> ByteArray, decodeImage: (ByteArray) -> HfImage.Rgb): Decider.Input = when (c.kind) {
        "text" -> Decider.Input.Text(c.input)
        "audio" -> Decider.Input.Audio(AudioIn.load(readFile(c.input)))
        "image" -> decodeImage(readFile(c.input)).let { Decider.Input.fromRgb(it.rgb, it.width, it.height) }
        else -> throw IllegalArgumentException("unknown kind ${c.kind}")
    }

    private fun median(x: List<Double>): Double = if (x.isEmpty()) Double.NaN else x.sorted().let { s -> if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2 }
    private fun maxAbsDiff(a: DoubleArray, b: DoubleArray) = a.indices.maxOf { abs(a[it] - b[it]) }
    private fun argmax(a: DoubleArray) = a.indices.maxByOrNull { a[it] }!!

    /** Runs every case [repeat] times (latency = median), returns the report. [only] limits to these case ids. */
    fun run(decider: BigModel, bundle: Bundle, readFile: (String) -> ByteArray, decodeImage: (ByteArray) -> HfImage.Rgb, repeat: Int = 1,
            only: Set<String>? = null, log: (String) -> Unit = {}): JSONObject {
        val rows = JSONArray(); val dps = mutableMapOf<String, MutableList<Double>>(); val ms = mutableMapOf<String, MutableList<Double>>()
        var promptMismatch = 0; var tokenMismatch = 0; var argmaxAgree = 0; var nParity = 0; val decoderDps = mutableListOf<Double>()
        val hfDps = mutableListOf<Double>()
        for (c in bundle.cases) {
            if (only != null && c.id !in only) continue
            val inp = input(c, readFile, decodeImage)
            val runs = (1..maxOf(1, repeat)).map { decider.evaluate(inp, c.question, c.options, c.descriptions, c.multi, c.rules, c.context) }
            val times = runs.map { it.ms.toDouble() }
            val res = runs.last()
            val dp = maxAbsDiff(res.p, c.expectedP)
            val promptOk = res.prompt == c.json.getString("prompt")
            val tokOk = res.positions == c.json.getInt("tokens")
            val agree = argmax(res.p) == argmax(c.expectedP)
            val t = median(times)
            val row = JSONObject().put("id", c.id).put("kind", c.kind).put("multi", c.multi).put("max_dp", dp).put("prompt_match", promptOk)
                .put("tokens", res.positions).put("tokens_expected", c.json.getInt("tokens")).put("ms", t).put("argmax_agree", agree)
                .put("p", JSONArray(res.p.toList())).put("p_expected", JSONArray(c.expectedP.toList())).put("z", JSONArray(res.z.toList()))
            c.hfP?.let { val d = maxAbsDiff(res.p, it); row.put("max_dp_vs_hf", d); if (!c.decoderRow) hfDps += d }
            if (c.decoderRow) { row.put("decoder_row", true); decoderDps += dp } else {
                nParity++; dps.getOrPut(c.kind) { mutableListOf() } += dp; ms.getOrPut(c.kind) { mutableListOf() } += t
                if (!promptOk) promptMismatch++; if (!tokOk) tokenMismatch++; if (agree) argmaxAgree++
            }
            rows.put(row)
            log("%s %-5s max|dp|=%.2e prompt=%s tok %d/%d %7.0f ms%s".format(c.id, c.kind, dp, if (promptOk) "ok" else "DIFF", res.positions,
                c.json.getInt("tokens"), t, if (c.decoderRow) "  (decoder row)" else ""))
        }
        val all = dps.values.flatten()
        val (hwm, rss) = memoryMb()
        val perKind = JSONObject()
        for ((k, v) in dps) perKind.put(k, JSONObject().put("n", v.size).put("median_dp", median(v)).put("max_dp", v.max()).put("median_ms", median(ms[k]!!)))
        val summary = JSONObject().put("n", nParity).put("median_dp", median(all)).put("max_dp", all.maxOrNull() ?: Double.NaN)
            .put("prompt_mismatches", promptMismatch).put("token_mismatches", tokenMismatch).put("argmax_agree", argmaxAgree)
            .put("per_kind", perKind).put("peak_rss_mb", hwm).put("rss_mb", rss).put("repeat", repeat)
            .put("reference", bundle.meta.optString("generated_by"))
        if (decoderDps.isNotEmpty()) summary.put("jpeg_decoder_rows", JSONObject().put("n", decoderDps.size).put("max_dp", decoderDps.max()))
        if (hfDps.isNotEmpty()) summary.put("vs_hf_bf16", JSONObject().put("n", hfDps.size).put("median_dp", median(hfDps)).put("max_dp", hfDps.max()))
        return JSONObject().put("summary", summary).put("rows", rows)
    }
}
