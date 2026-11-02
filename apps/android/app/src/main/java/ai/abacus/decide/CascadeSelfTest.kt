package ai.abacus.decide

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

/**
 * Replays the cascade bundle (assets/cascade_selftest/expected.json, written by fast/make_cascade_bundle.py) through [Cascade]:
 * per request, the route must equal the Python reference's route (decided on the same ONNX fast-tier path with
 * smaug_decide.cascade.confidence), the fast-tier probabilities are compared with that Python replay (p_onnx) and with PyTorch
 * (p_torch), and Smaug-Decisions' probabilities (escalated or routed requests) with llama_decide.py through the same C core.
 * Also reports Smaug-Decisions loads, memory before / after Smaug-Decisions, and latencies. No Android dependencies (shared by the desktop host test and
 * the device self-test); the caller supplies file reading and image decoding.
 */
object CascadeSelfTest {
    fun doubles(a: JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }
    private fun maxAbs(a: DoubleArray, b: DoubleArray) = a.indices.maxOf { abs(a[it] - b[it]) }
    private fun median(x: List<Double>) = if (x.isEmpty()) Double.NaN else x.sorted().let { s -> if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2 }

    fun run(cascade: Cascade, bundle: JSONObject, scenarios: Map<String, Scenario>, readFile: (String) -> ByteArray,
            decodeImage: (ByteArray) -> HfImage.Rgb, log: (String) -> Unit = {}, bigKey: String = "p_big"): JSONObject {
        val reqs = bundle.getJSONArray("requests"); val rows = JSONArray()
        val fastVsOnnx = mutableListOf<Double>(); val fastVsTorch = mutableListOf<Double>(); val bigVsPy = mutableListOf<Double>()
        val msFast = mutableListOf<Double>(); val msBig = mutableListOf<Double>()
        var routeMismatch = 0; var routeTorchFlips = 0; var promptMismatch = 0; var posMismatch = 0
        val (hwm0, _) = SelfTest.memoryMb(); var rssFastOnly = -1.0; var splitFastOnly = JSONObject()
        for (i in 0 until reqs.length()) {
            val q = reqs.getJSONObject(i); val s = scenarios.getValue(q.getString("scenario"))
            val req = when (q.getString("kind")) {
                "audio" -> Cascade.Request.Audio(s, AudioIn.load(readFile(q.getString("input"))))
                "text" -> Cascade.Request.Text(s, q.getString("input"))
                else -> decodeImage(readFile(q.getString("input"))).let { Cascade.Request.Image(s, Decider.Input.fromRgb(it.rgb, it.width, it.height)) }
            }
            if (!cascade.bigLoaded) { rssFastOnly = SelfTest.memoryMb().first; splitFastOnly = SelfTest.rssSplitMb() }   // fast tier only so far
            val r = cascade.decide(req)
            val expRoute = q.getString("route")
            val got = when (r.route) { Cascade.Route.FAST -> "fast"; Cascade.Route.ESCALATED -> "escalated"; Cascade.Route.ROUTED -> "routed"; Cascade.Route.FLAGGED -> "flagged"; Cascade.Route.UNSURE -> "unsure"; Cascade.Route.NOT_INSTALLED -> "not_installed" }
            val row = JSONObject().put("id", q.getString("id")).put("route", got).put("route_expected", expRoute)
            if (got != expRoute) routeMismatch++
            if (q.has("route_torch") && q.getString("route_torch") != expRoute) routeTorchFlips++
            r.fastP?.let { fp ->
                val dO = maxAbs(fp, doubles(q.getJSONArray("p_onnx"))); val dT = maxAbs(fp, doubles(q.getJSONArray("p_torch")))
                fastVsOnnx += dO; fastVsTorch += dT; msFast += r.msFast
                row.put("fast_p", JSONArray(fp.toList())).put("fast_conf", r.fastConfidence).put("fast_dp_vs_python_onnx", dO).put("fast_dp_vs_torch", dT)
                    .put("conf_expected", q.getDouble("conf_onnx")).put("ms_fast", r.msFast)
            }
            if (r.bigP != null && q.has(bigKey)) {
                val d = maxAbs(r.bigP, doubles(q.getJSONArray(bigKey))); bigVsPy += d; msBig += r.msBig
                row.put("big_p", JSONArray(r.bigP.toList())).put("big_dp_vs_python", d).put("ms_big", r.msBig).put("big_load_ms", r.bigLoadMs)
            }
            row.put("mem", SelfTest.smapsMb())
            rows.put(row)
            log("%-34s %-9s (expected %-9s) %s".format(q.getString("id"), got, expRoute,
                listOfNotNull(row.optDouble("fast_dp_vs_python_onnx").takeIf { !it.isNaN() }?.let { "fast |dp| %.1e".format(it) },
                    row.optDouble("big_dp_vs_python").takeIf { !it.isNaN() }?.let { "Smaug-Decisions |dp| %.1e".format(it) }).joinToString(", ")))
        }
        val (hwm, rss) = SelfTest.memoryMb()
        // peak smaps_rollup values after each kind of request (one process: audio requests come first, then text, then the image)
        val memByKind = JSONObject()
        for (i in 0 until rows.length()) {
            val row = rows.getJSONObject(i); val q = reqs.getJSONObject(i)
            val key = when { q.getString("kind") != "audio" -> q.getString("kind"); row.getString("route") == "fast" -> "audio_fast"; else -> "audio_big" }
            val m = row.getJSONObject("mem"); val acc = memByKind.optJSONObject(key) ?: JSONObject().also { memByKind.put(key, it) }
            for (k in listOf("Rss", "Pss", "Pss_Anon", "Pss_File", "Private_Dirty", "Anonymous"))
                if (m.has(k)) acc.put(k, maxOf(acc.optDouble(k, 0.0), m.getDouble(k)))
        }
        val summary = JSONObject().put("n", reqs.length()).put("route_mismatches", routeMismatch).put("route_flips_torch_vs_onnx_in_bundle", routeTorchFlips)
            .put("escalated", (0 until rows.length()).count { rows.getJSONObject(it).getString("route") == "escalated" })
            .put("routed", (0 until rows.length()).count { rows.getJSONObject(it).getString("route") == "routed" })
            .put("fast_answered", (0 until rows.length()).count { rows.getJSONObject(it).getString("route") == "fast" })
            .put("fast_vs_python_onnx", JSONObject().put("n", fastVsOnnx.size).put("median_dp", median(fastVsOnnx)).put("max_dp", fastVsOnnx.maxOrNull() ?: Double.NaN))
            .put("fast_vs_torch", JSONObject().put("n", fastVsTorch.size).put("median_dp", median(fastVsTorch)).put("max_dp", fastVsTorch.maxOrNull() ?: Double.NaN))
            .put("big_vs_python_llama", JSONObject().put("n", bigVsPy.size).put("median_dp", median(bigVsPy)).put("max_dp", bigVsPy.maxOrNull() ?: Double.NaN))
            .put("median_ms_fast", median(msFast)).put("median_ms_big", median(msBig)).put("big_loads", cascade.bigLoads).put("big_load_ms", cascade.lastBigLoadMs)
            .put("peak_rss_mb_before", hwm0).put("peak_rss_mb_fast_only", rssFastOnly).put("peak_rss_mb_end", hwm).put("rss_mb_end", rss).put("tau", cascade.tau)
            .put("rss_split_fast_only", splitFastOnly).put("smaps_peak_after_kind_mb", memByKind).put("rss_split_end", SelfTest.rssSplitMb())
        return JSONObject().put("summary", summary).put("rows", rows)
    }
}
