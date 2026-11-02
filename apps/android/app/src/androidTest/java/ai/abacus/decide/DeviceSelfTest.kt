package ai.abacus.decide

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device parity + latency + memory: replays assets/selftest through the app's Decider (real GGUF, pushed by
 * run_device_test.sh). Instrumentation args (-e): threads, repeat, only (comma-separated case ids), max_dp (default 0.1:
 * arm64 and x86 Q4 kernels differ by a few 1e-2; prompts and positions must match exactly).
 */
@RunWith(AndroidJUnit4::class)
class DeviceSelfTest {
    @Test fun replayReferenceCases() {
        val inst = InstrumentationRegistry.getInstrumentation(); val args = InstrumentationRegistry.getArguments()
        val ctx = inst.targetContext
        val threads = args.getString("threads")?.toInt() ?: AndroidSelfTest.defaultThreads()
        val repeat = args.getString("repeat")?.toInt() ?: 1
        val only = args.getString("only")?.split(",")?.map { it.trim() }?.toSet()
        val maxDp = args.getString("max_dp")?.toDouble() ?: 0.1
        val report = AndroidSelfTest.run(ctx, threads = threads, repeat = repeat, only = only) { Log.i("decide-selftest", it) }
        val s = report.getJSONObject("summary")
        Log.i("decide-selftest", "SUMMARY " + s.toString())
        assertEquals("rendered prompts differ from Python", 0, s.getInt("prompt_mismatches"))
        assertEquals("prompt positions differ from Python", 0, s.getInt("token_mismatches"))
        assertTrue("max |dp| vs Python ${s.getDouble("max_dp")} >= $maxDp", s.getDouble("max_dp") < maxDp)
    }
}
