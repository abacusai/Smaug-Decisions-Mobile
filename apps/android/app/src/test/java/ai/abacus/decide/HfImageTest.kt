package ai.abacus.decide

import ai.abacus.decide.Goldens.objects
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** HF grid sizes and the Pillow-bicubic port vs Python (llama_decide.hf_image_size / hf_resize / Image.resize). */
class HfImageTest {
    @Test fun gridSizesAndTokenCountsMatchPython() {
        val cases = Goldens.array("image_sizes.json").objects(); assertTrue(cases.size >= 60)
        for (c in cases) {
            val (th, tw) = HfImage.size(c.getInt("h"), c.getInt("w"))
            val at = "${c.getInt("h")}x${c.getInt("w")}"
            assertEquals(at, c.getInt("th"), th); assertEquals(at, c.getInt("tw"), tw)
            assertEquals(at, c.getInt("tokens"), HfImage.softTokens(th, tw))
        }
    }

    @Test fun bicubicResizeIsBitExactWithPillow() {
        val cases = Goldens.array("resize.json").objects(); assertTrue(cases.size >= 30)
        for (c in cases) {
            val src = loadPng("images/${c.getString("name")}.png")
            assertEquals(c.getInt("w"), src.width); assertEquals(c.getInt("h"), src.height)
            val out = if (c.optBoolean("plain")) HfImage.resizeBicubic(src, c.getInt("tw"), c.getInt("th")) else HfImage.toGrid(src)
            val at = "${c.getString("name")} -> ${c.getInt("tw")}x${c.getInt("th")}"
            assertEquals(at, c.getInt("tw"), out.width); assertEquals(at, c.getInt("th"), out.height)
            assertEquals(at, c.getString("sha256"), Goldens.sha256(out.rgb))
        }
    }

    companion object {
        fun loadPng(name: String): HfImage.Rgb = JvmImages.decode(Goldens.bytes(name))
    }
}
