package ai.abacus.decide

import org.json.JSONArray
import org.json.JSONObject

/**
 * One question the app can ask (fast/scenarios.py, pre-tokenised by fast/make_scenarios.py so the phone needs no tokenizer for the audio model).
 * Audio scenarios carry the audio model's prompt split around the audio block: [preIds] (up to and including <audio_bos>) and [postIds]
 * (from <audio_eos>), with the option markers' offsets in post; the app inserts one <|audio|> id per audio token. Every scenario
 * also carries the plain question / options / descriptions / rules / text that Smaug-Decisions renders with [Prompt] when it escalates.
 */
class Scenario(val json: JSONObject, val audioTokenId: Int) {
    val id: String = json.getString("id")
    val title: String = json.optString("title", id)
    val kind: String = json.getString("kind")
    val question: String = json.getString("question")
    val multi: Boolean = json.optBoolean("multi", false)
    val options: List<String> = strings(json.getJSONArray("options"))
    val descriptions: List<String?> = json.optJSONArray("descriptions")?.let { stringsOrNull(it) } ?: List(options.size) { null }
    val rules: String? = json.optString("rules").takeIf { json.has("rules") && !json.isNull("rules") }
    val text: String? = json.optString("text").takeIf { json.has("text") && !json.isNull("text") }
    val preIds: IntArray = ints(json.optJSONArray("pre_ids"))
    val postIds: IntArray = ints(json.optJSONArray("post_ids"))
    val markerOffsetsInPost: List<Int> = ints(json.optJSONArray("marker_offsets_in_post")).toList()

    companion object {
        fun strings(a: JSONArray) = (0 until a.length()).map { a.getString(it) }
        fun stringsOrNull(a: JSONArray) = (0 until a.length()).map { if (a.isNull(it)) null else a.getString(it) }
        fun ints(a: JSONArray?) = if (a == null) IntArray(0) else IntArray(a.length()) { a.getInt(it) }

        /** scenarios.json -> scenarios (with the file's <|audio|> id) */
        fun parseFile(text: String): List<Scenario> {
            val o = JSONObject(text); val aud = o.getInt("audio_token_id"); val a = o.getJSONArray("scenarios")
            return (0 until a.length()).map { Scenario(a.getJSONObject(it), aud) }
        }
    }
}
