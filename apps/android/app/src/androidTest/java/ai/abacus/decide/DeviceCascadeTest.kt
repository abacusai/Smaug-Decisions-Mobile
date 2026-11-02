package ai.abacus.decide

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device cascade: assets/cascade_selftest through the app's own Cascade (fast tier memory-mapped from the APK, Smaug-Decisions from the
 * external files dir) with the device's real tier. On a FULL / STANDARD / SQUEEZE device with Smaug-Decisions installed (and enough free memory) the routes must equal the Python
 * reference; on a FAST_ONLY device (or without Smaug-Decisions) the fast answers must match and the rest must be flagged. Writes
 * cascade-report.json next to the models. -e max_dp (default 0.1: arm64 vs x86 int8 / Q4 kernel noise).
 */
@RunWith(AndroidJUnit4::class)
class DeviceCascadeTest {
    @Test fun replayCascade() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val maxDp = InstrumentationRegistry.getArguments().getString("max_dp")?.toDouble() ?: 0.1
        val tier = AndroidCascade.deviceTier(ctx)
        AndroidCascade.cascade(ctx, AndroidCascade.fastTier(ctx), tier = tier).use { c ->
            val r = AndroidCascade.selfTest(ctx, c) { Log.i("decide-cascade", it) }
            val s = r.getJSONObject("summary")
            Log.i("decide-cascade", "SUMMARY tier=$tier " + s.toString())
            assertTrue("fast tier vs Python", s.getJSONObject("fast_vs_python_onnx").getDouble("max_dp") < maxDp)
            if (tier != DeviceTier.FAST_ONLY && AndroidCascade.bigInstalled(ctx) && s.getInt("routed") > 0) {
                // routes can differ only where the fast confidence sits within kernel noise of tau
                assertTrue("route mismatches ${s.getInt("route_mismatches")}", s.getInt("route_mismatches") <= 2)
                assertTrue("Smaug-Decisions vs Python", s.getJSONObject("big_vs_python_llama").getDouble("max_dp") < maxDp)
            } else assertEquals(0, c.bigLoads)
        }
    }
}
