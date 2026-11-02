package ai.abacus.decide

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The whole cascade on the desktop JVM with the real models: the app's FastTier (ONNX Runtime desktop jar, same Java API as
 * onnxruntime-android) + Cascade + Decider -> JNI -> decide_core -> llama.cpp (host build), over the cascade bundle
 * (assets/cascade_selftest). Asserts the routes equal the Python reference, fast-tier probabilities equal the Python replay of the
 * same ONNX path, and Smaug-Decisions probabilities equal llama_decide.py. Then checks release / reload and the fast-only policy.
 * Needs -Pdecide.fastDir (ONNX units) and, for Smaug-Decisions, -Pdecide.hostLib / -Pdecide.model / -Pdecide.mmproj (see run_cascade_host.sh).
 */
class CascadeHostTest {
    private val assets = File(System.getProperty("decide.assets") ?: "src/main/assets")

    private fun scenarios() = Scenario.parseFile(File(assets, "fast/scenarios.json").readText()).associateBy { it.id }

    @Test fun cascadeMatchesPython() {
        val fastDir = System.getProperty("decide.fastDir")
        assumeTrue("cascade host test skipped: pass -Pdecide.fastDir, -Pdecide.hostLib and the Smaug-Decisions model (see TestBig)",
            fastDir != null && TestBig.configured)
        val bigKey = if (TestBig.backend == "litert") "p_big_litert" else "p_big"
        val dir = File(System.getProperty("decide.cascadeBundle") ?: File(assets, "cascade_selftest").path)
        val bundle = JSONObject(File(dir, "expected.json").readText())
        val threads = System.getProperty("decide.threads")?.toInt() ?: 4
        val (hwm0, _) = SelfTest.memoryMb()
        val t0 = System.nanoTime()
        val fast = FastTier(File(fastDir), threads)
        val fastLoadS = (System.nanoTime() - t0) / 1e9
        val (hwmFastLoaded, rssFastLoaded) = SelfTest.memoryMb()
        val cascade = Cascade(fast, { TestBig.create(bundle.getDouble("b"), threads) }, tau = bundle.getDouble("tau"))
        val report = cascade.use { c ->
            val r = CascadeSelfTest.run(c, bundle, scenarios(), { File(dir, it).readBytes() }, JvmImages::decode, ::println, bigKey)
            // memory pressure: release Smaug-Decisions, the next escalation must reload it and give the same answer
            val esc = (0 until bundle.getJSONArray("requests").length()).map { bundle.getJSONArray("requests").getJSONObject(it) }
                .first { it.getString("route") == "escalated" }
            c.releaseBig(); assertTrue(!c.bigLoaded)
            val s = scenarios().getValue(esc.getString("scenario"))
            val again = c.decide(Cascade.Request.Audio(s, AudioIn.load(File(dir, esc.getString("input")).readBytes())))
            assertEquals(Cascade.Route.ESCALATED, again.route)
            val dReload = again.bigP!!.indices.maxOf { kotlin.math.abs(again.bigP!![it] - esc.getJSONArray(bigKey).getDouble(it)) }
            r.getJSONObject("summary").put("reload_after_release", JSONObject().put("big_loads", c.bigLoads).put("max_dp", dReload).put("load_ms", c.lastBigLoadMs))
                .put("fast_load_s", fastLoadS).put("peak_rss_mb_jvm_start", hwm0).put("peak_rss_mb_after_fast_load", hwmFastLoaded)
                .put("rss_mb_after_fast_load", rssFastLoaded).put("threads", threads).put("platform", "desktop JVM " + System.getProperty("os.arch")).put("backend", TestBig.backend)
            r
        }
        val out = File(System.getProperty("decide.report") ?: "build/cascade-host-report.json"); out.parentFile?.mkdirs(); out.writeText(report.toString(1))
        val s = report.getJSONObject("summary")
        println("CASCADE HOST SUMMARY " + s.toString(1) + "\nreport: " + out.absolutePath)
        assertEquals("routes differ from the Python reference", 0, s.getInt("route_mismatches"))
        assertTrue("fast tier vs Python ONNX replay", s.getJSONObject("fast_vs_python_onnx").getDouble("max_dp") < 1e-3)
        assertTrue("Smaug-Decisions vs llama_decide.py", s.getJSONObject("big_vs_python_llama").getDouble("max_dp") < 1e-3)
        assertTrue("reload after release", s.getJSONObject("reload_after_release").getDouble("max_dp") < 1e-3)
    }

    /** Fast-only device: no Smaug-Decisions is ever constructed; unsure audio decisions are flagged, text/image need Smaug-Decisions. Fast tier only. */
    @Test fun fastOnlyPolicyNeverLoadsBig() {
        val fastDir = System.getProperty("decide.fastDir")
        assumeTrue("needs -Pdecide.fastDir", fastDir != null)
        val dir = File(System.getProperty("decide.cascadeBundle") ?: File(assets, "cascade_selftest").path)
        val bundle = JSONObject(File(dir, "expected.json").readText())
        val reqs = bundle.getJSONArray("requests"); val sc = scenarios()
        Cascade(FastTier(File(fastDir), 4), { error("Smaug-Decisions must not load on a fast-only device") }, bundle.getDouble("tau"), DeviceTier.FAST_ONLY).use { c ->
            var flagged = 0; var fastN = 0
            for (i in 0 until reqs.length()) {
                val q = reqs.getJSONObject(i); val s = sc.getValue(q.getString("scenario"))
                val r = when (q.getString("kind")) {
                    "audio" -> c.decide(Cascade.Request.Audio(s, AudioIn.load(File(dir, q.getString("input")).readBytes())))
                    "text" -> c.decide(Cascade.Request.Text(s, q.getString("input")))
                    else -> continue
                }
                val expectFast = q.getString("route") == "fast"
                assertEquals(q.getString("id"), if (expectFast) Cascade.Route.FAST else Cascade.Route.FLAGGED, r.route)
                if (r.route == Cascade.Route.FLAGGED) flagged++ else fastN++
            }
            assertEquals(0, c.bigLoads)
            println("FAST-ONLY: $fastN answered on the fast tier, $flagged flagged 'would escalate' / needs Smaug-Decisions; memory after them: " +
                SelfTest.rssSplitMb())
        }
    }
}
