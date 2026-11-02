package ai.abacus.decide

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import kotlin.math.abs

/**
 * Fast tier, pure-Kotlin parts vs Python (goldens/mel.json from fast/make_fast_goldens.py): the log-mel front end vs transformers'
 * Gemma4AudioFeatureExtractor (valid frames, frame counts, audio-token counts), fp16 -> fp32 conversion of the embedding table,
 * and the cascade confidence rule vs smaug_decide.cascade.confidence. Also the device-tier thresholds.
 */
class FastFrontEndTest {
    private val g = Goldens.obj("mel.json")

    private fun floats(b64: String): FloatArray {
        val bb = ByteBuffer.wrap(Base64.getDecoder().decode(b64)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bb.remaining()).also { bb.get(it) }
    }

    @Test fun melMatchesTransformers() {
        val clips = g.getJSONArray("clips"); var worst = 0.0
        for (i in 0 until clips.length()) {
            val c = clips.getJSONObject(i)
            val pcm = AudioIn.load(Goldens.bytes(c.getString("wav")))
            assertEquals(c.getString("name"), c.getInt("samples_used"), pcm.size)
            val frames = MelFrontEnd.frameCount(pcm.size)
            assertEquals(c.getString("name"), c.getInt("frames"), frames)
            if (frames == 0) continue
            assertEquals(c.getString("name"), c.getInt("audio_tokens"), MelFrontEnd.audioTokens(frames))
            val k = MelFrontEnd.features(pcm); val ref = floats(c.getString("features_b64"))
            assertEquals(ref.size, k.size)
            val d = k.indices.maxOf { abs(k[it] - ref[it]).toDouble() }
            println("mel %-14s %5d frames max|diff| %.2e".format(c.getString("name"), frames, d))
            worst = maxOf(worst, d)
        }
        println("mel worst max|diff| vs transformers: %.2e".format(worst))
        // the residual sits at bins near the log floor (log(1e-3) = -6.9), where transformers' float32 FFT noise dominates: the float64
        // numpy spec (fast/mel.py) shows the same 3.2e-4 on chirp_1s
        assertTrue("mel max |diff| $worst", worst < 5e-4)
    }

    @Test fun halfToFloatExact() {
        val bits = g.getJSONArray("half_bits"); val vals = g.getJSONArray("half_values")
        for (i in 0 until bits.length()) assertEquals(vals.getDouble(i).toFloat(), Half.toFloat(bits.getInt(i).toShort()), 0f)
    }

    @Test fun confidenceMatchesPython() {
        val a = g.getJSONArray("confidence"); var worst = 0.0
        for (i in 0 until a.length()) {
            val c = a.getJSONObject(i); val p = c.getJSONArray("p").let { x -> DoubleArray(x.length()) { x.getDouble(it) } }
            worst = maxOf(worst, abs(Cascade.confidence(p, c.getBoolean("multi")) - c.getDouble("confidence")))
        }
        assertTrue("confidence max diff $worst", worst <= 1e-15)
    }

    @Test fun deviceTiers() {
        val gib = 1024L * 1024 * 1024; val mb = 1024L * 1024
        assertEquals(DeviceTier.FAST_ONLY, DevicePolicy.tier((3.6 * gib).toLong()))   // 4 GB phone
        assertEquals(DeviceTier.SQUEEZE, DevicePolicy.tier((5.5 * gib).toLong()))     // 6 GB phone
        assertEquals(DeviceTier.STANDARD, DevicePolicy.tier((7.4 * gib).toLong()))    // 8 GB phone
        assertEquals(DeviceTier.FULL, DevicePolicy.tier((11.2 * gib).toLong()))       // 12 GB phone
        assertEquals(DeviceTier.FULL, DevicePolicy.tier((15.0 * gib).toLong()))       // 16 GB phone
        assertTrue(DevicePolicy.canUseBig(DeviceTier.STANDARD, "audio", 2500 * mb, false))
        assertTrue(!DevicePolicy.canUseBig(DeviceTier.STANDARD, "audio", 1500 * mb, false))
        assertTrue(DevicePolicy.canUseBig(DeviceTier.STANDARD, "audio", 1500 * mb, true))
        assertTrue(!DevicePolicy.canUseBig(DeviceTier.SQUEEZE, "image", 3000 * mb, true))
        assertTrue(DevicePolicy.canUseBig(DeviceTier.FULL, "image", 500 * mb, false))
        assertTrue(!DevicePolicy.canUseBig(DeviceTier.FAST_ONLY, "text", 16 * gib, false))
        assertTrue(DevicePolicy.keepVision(DeviceTier.FULL) && !DevicePolicy.keepVision(DeviceTier.STANDARD))
    }
}
