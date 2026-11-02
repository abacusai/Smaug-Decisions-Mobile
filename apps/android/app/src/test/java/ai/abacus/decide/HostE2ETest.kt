package ai.abacus.decide

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * End to end on the desktop JVM: the app's own Decider -> JNI (decide_jni.c) -> decide_core.c -> llama.cpp, built for the host
 * (libdecide_jni.so on java.library.path), real GGUF on CPU, over the self-test bundle; compared with the Python path's expected
 * probabilities. Skipped unless -Pdecide.hostLib / -Pdecide.model / -Pdecide.mmproj are given (see run_host_e2e.sh).
 */
class HostE2ETest {
    @Test fun kotlinPathMatchesPython() {
        assumeTrue("host e2e skipped: pass -Pdecide.hostLib and the Smaug-Decisions model (see TestBig)", TestBig.configured)
        val dir = File(System.getProperty("decide.bundle"))
        val bundle = SelfTest.parse(File(dir, "expected.json").readText())
        val threads = System.getProperty("decide.threads")?.toInt() ?: bundle.meta.optInt("threads", 4)
        val only = System.getProperty("decide.only")?.split(",")?.map { it.trim() }?.toSet()
        val smaps0 = SelfTest.smapsMb()
        val t0 = System.nanoTime()
        val report = TestBig.create(bundle.b, threads).use { d ->
            val loadS = (System.nanoTime() - t0) / 1e9
            println("loaded in %.1f s, %d threads".format(loadS, threads))
            SelfTest.run(d, bundle, { File(dir, it).readBytes() }, JvmImages::decode, only = only, log = ::println).also {
                it.getJSONObject("summary").put("load_s", loadS).put("threads", threads).put("platform", "desktop JVM " + System.getProperty("os.arch")).put("backend", d.backend).put("smaps_end_mb", SelfTest.smapsMb()).put("smaps_before_load_mb", smaps0)
            }
        }
        val out = File(System.getProperty("decide.report") ?: "build/host-e2e-report.json")
        out.parentFile?.mkdirs(); out.writeText(report.toString(1))
        val s = report.getJSONObject("summary")
        println("HOST E2E SUMMARY " + s.toString(1) + "\nreport: " + out.absolutePath)
        assertEquals("rendered prompts differ from Python", 0, s.getInt("prompt_mismatches"))
        assertEquals("prompt positions differ from Python", 0, s.getInt("token_mismatches"))
        assertTrue("max |dp| vs Python = ${s.getDouble("max_dp")}", s.getDouble("max_dp") < 1e-3)
    }
}
