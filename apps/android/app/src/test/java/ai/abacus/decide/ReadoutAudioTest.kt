package ai.abacus.decide

import ai.abacus.decide.Goldens.doubles
import ai.abacus.decide.Goldens.objects
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Readout math (softmax / sigmoid(z - b)) and audio loading (WAV decode, np.interp resample, last 30 s) vs numpy / soundfile. */
class ReadoutAudioTest {
    @Test fun readoutMatchesNumpy() {
        val g = Goldens.obj("readout.json"); val b = g.getDouble("b")
        assertEquals(Decider.DEFAULT_B, b, 0.0)
        var maxErr = 0.0
        val cases = g.getJSONArray("cases").objects(); assertTrue(cases.size >= 200)
        for (c in cases) {
            val z = c.getJSONArray("z").doubles()
            val sm = Readout.probabilities(z, false, b); val sg = Readout.probabilities(z, true, b)
            val wantSm = c.getJSONArray("softmax").doubles(); val wantSg = c.getJSONArray("sigmoid").doubles()
            for (i in z.indices) { maxErr = maxOf(maxErr, kotlin.math.abs(sm[i] - wantSm[i]), kotlin.math.abs(sg[i] - wantSg[i])) }
            assertEquals(1.0, sm.sum(), 1e-12)
        }
        assertEquals("max |p_kotlin - p_numpy|", 0.0, maxErr, 1e-14)
    }

    @Test fun wavLoadingMatchesSoundfileAndLoadAudio() {
        val cases = Goldens.obj("audio.json").getJSONArray("wav").objects(); assertTrue(cases.size >= 8)
        for (c in cases) {
            val f = c.getString("file")
            val w = AudioIn.readWav(Goldens.bytes("audio/$f"))
            assertEquals(f, c.getInt("sr"), w.sampleRate)
            val out = AudioIn.load(Goldens.bytes("audio/$f"))
            assertEquals(f, c.getInt("n_out"), out.size)
            val head = c.getJSONArray("head").doubles()
            for (i in head.indices) assertEquals("$f[$i]", head[i].toFloat(), out[i], 0f)
            assertEquals(f, c.getString("sha256"), Goldens.sha256(Goldens.floatsLE(out)))
        }
    }

    @Test fun resampleMatchesNpInterp() {
        val cases = Goldens.obj("audio.json").getJSONArray("resample").objects(); assertTrue(cases.size >= 8)
        for (c in cases) {
            val x = Goldens.floatsFromB64(c.getString("x")); val want = Goldens.floatsFromB64(c.getString("y"))
            val got = AudioIn.lastSeconds(AudioIn.resampleTo16k(x, c.getInt("sr")))
            assertArrayEquals("sr=${c.getInt("sr")}", want, got, 0f)
        }
    }
}
