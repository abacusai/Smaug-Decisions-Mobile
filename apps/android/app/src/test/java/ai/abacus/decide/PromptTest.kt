package ai.abacus.decide

import ai.abacus.decide.Goldens.nullableStrings
import ai.abacus.decide.Goldens.objects
import ai.abacus.decide.Goldens.optStr
import ai.abacus.decide.Goldens.strings
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Prompt.render / trim / char budget vs the Python reference render (goldens from make_kotlin_goldens.py). */
class PromptTest {
    @Test fun renderMatchesPythonByteForByte() {
        val cases = Goldens.array("render.json").objects()
        assertTrue("expected ~45 render goldens", cases.size >= 40)
        val kinds = mutableSetOf<String>()
        for (c in cases) {
            val options = c.getJSONArray("options").strings()
            val media = c.getString("media")
            val audioSamples = if (media == "audio") c.getInt("audio_samples") else null
            val budget = Prompt.charBudget(Prompt.mediaTokens(audioSamples, media == "image"), options.size)
            assertEquals("budget: ${c.getString("note")}", c.getInt("budget"), budget)
            val desc = if (c.isNull("descriptions")) List(options.size) { null } else c.getJSONArray("descriptions").nullableStrings()
            val tag = when (media) { "audio" -> Prompt.AUDIO_TAG; "image" -> Prompt.IMAGE_TAG; else -> null }
            val prompt = Prompt.render(c.getString("question"), options, desc, c.getBoolean("multi"), tag, c.optStr("rules"),
                c.optStr("context"), if (tag == null) c.optStr("text") else null, budget)
            val want = c.getString("prompt")
            assertEquals(c.getString("note"), want, prompt)
            assertArrayEquals(c.getString("note"), want.toByteArray(Charsets.UTF_8), prompt.toByteArray(Charsets.UTF_8))
            kinds += media + if (c.getBoolean("multi")) "-multi" else if (options == listOf("Yes", "No")) "-predicate" else "-choice"
        }
        assertTrue(kinds.toString(), kinds.containsAll(listOf("text-multi", "text-predicate", "text-choice", "audio-multi", "image-multi", "image-choice")))
    }

    @Test fun trimMatchesPython() {
        val cases = Goldens.array("trim.json").objects(); assertTrue(cases.size >= 80)
        for (c in cases) assertEquals("n=${c.getInt("n")}", c.getString("out"), Prompt.trim(c.getString("s"), c.getInt("n")))
    }
}
