package ai.abacus.decide

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * The shipped int8 token table (+ per-row scales) against the fp16 table it was made from, through the app's FastTier on the
 * desktop JVM: every audio request of the cascade bundle (assets/cascade_selftest) and, with -Pdecide.parityClips=<dir with
 * cases.jsonl + wavs> (fast/make_clips.py), the 100-decision parity set. Same ONNX units, only the table differs.
 * Needs -Pdecide.fastDir with audio.onnx, trunk.onnx, head.json, embed_tokens_int8.bin, embed_tokens_int8_scales.bin and
 * embed_tokens_f16.bin.
 */
class EmbedInt8ParityTest {
    private val assets = File(System.getProperty("decide.assets") ?: "src/main/assets")

    @Test fun int8TableMatchesFp16() {
        val fastDir = System.getProperty("decide.fastDir")?.let(::File)
        assumeTrue("needs -Pdecide.fastDir with both tables", fastDir != null && File(fastDir, EmbedTable.INT8).exists() && File(fastDir, EmbedTable.FP16).exists())
        val fp16 = EmbedTable.fp16(EmbedTable.map(File(fastDir, EmbedTable.FP16)))
        val sc = Scenario.parseFile(File(assets, "fast/scenarios.json").readText()).associateBy { it.id }
        val tau = Cascade.DEFAULT_TAU
        val cases = mutableListOf<Triple<String, Scenario, File>>()
        val bundleDir = File(assets, "cascade_selftest")
        val reqs = JSONObject(File(bundleDir, "expected.json").readText()).getJSONArray("requests")
        for (i in 0 until reqs.length()) reqs.getJSONObject(i).takeIf { it.getString("kind") == "audio" }?.let {
            cases += Triple("bundle:" + it.getString("id"), sc.getValue(it.getString("scenario")), File(bundleDir, it.getString("input")))
        }
        System.getProperty("decide.parityClips")?.let(::File)?.let { dir ->
            File(dir, "cases.jsonl").readLines().filter { it.isNotBlank() }.map(::JSONObject).forEach { c ->
                val opts = c.getJSONArray("options").let { a -> List(a.length()) { a.getString(it) } }
                val s = sc.values.firstOrNull { it.kind == "audio" && it.question == c.getString("question") && it.options == opts } ?: return@forEach
                cases += Triple("clips:" + c.getString("id"), s, File(dir, c.getString("input")))
            }
        }
        val rows = JSONArray(); val dps = mutableListOf<Double>(); var topSame = 0; var routeSame = 0; var nonTieFlips = 0
        FastTier(fastDir!!, 4).use { fast ->
            assertEquals("int8", fast.table.format)
            for ((id, s, wav) in cases) {
                val pcm = AudioIn.load(wav.readBytes())
                val a = fast.run(s, pcm); val b = fast.run(s, pcm, fp16)
                val dp = a.p.indices.maxOf { abs(a.p[it] - b.p[it]) }
                val same = if (s.multi) a.p.indices.all { (a.p[it] > 0.5) == (b.p[it] > 0.5) } else a.p.indices.maxBy { a.p[it] } == b.p.indices.maxBy { b.p[it] }
                val ca = Cascade.confidence(a.p, s.multi); val cb = Cascade.confidence(b.p, s.multi)
                if (!same && !s.multi) { val top2 = b.p.sortedDescending(); if (top2[0] - top2[1] > 0.05) nonTieFlips++ }
                dps += dp; if (same) topSame++; if ((ca >= tau) == (cb >= tau)) routeSame++
                rows.put(JSONObject().put("id", id).put("max_dp", dp).put("p_int8", JSONArray(a.p.toList())).put("p_fp16", JSONArray(b.p.toList()))
                    .put("conf_int8", ca).put("conf_fp16", cb))
            }
        }
        val sorted = dps.sorted()
        val summary = JSONObject().put("n", dps.size).put("max_dp", sorted.last()).put("median_dp", sorted[sorted.size / 2])
            .put("p90_dp", sorted[(sorted.size * 9) / 10]).put("top_answer_same", topSame).put("route_same_at_tau", routeSame).put("top_flips_outside_near_ties", nonTieFlips).put("tau", tau)
            .put("n_bundle", cases.count { it.first.startsWith("bundle:") }).put("n_clips", cases.count { it.first.startsWith("clips:") })
        val out = File(System.getProperty("decide.report") ?: "build/embed-int8-parity.json"); out.parentFile?.mkdirs()
        out.writeText(JSONObject().put("summary", summary).put("rows", rows).toString(1))
        println("EMBED INT8 PARITY " + summary.toString(1) + "\nreport: " + out.absolutePath)
        // Measured 2026-10-30 (132 decisions): median 0.0039, p90 0.0095, max 0.032 (a 0.50 / 0.50 tie); the shipped ONNX
        // quantisation itself is median 0.0076, max 0.063 vs PyTorch. Routes must agree; a top answer may only flip on a near-tie.
        assertTrue("int8 vs fp16 max |dp| ${sorted.last()}", sorted.last() < 0.05)
        assertEquals("routes at tau", dps.size, routeSame)
        assertEquals("top-answer flips outside near-ties", 0, nonTieFlips)
    }
}
