package ai.abacus.decide

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Goldens written by llama.cpp/make_kotlin_goldens.py from the Python reference (test resources). */
object Goldens {
    fun bytes(name: String): ByteArray =
        requireNotNull(Goldens::class.java.classLoader!!.getResourceAsStream("goldens/$name")) { "missing test resource goldens/$name" }.use { it.readBytes() }

    fun text(name: String) = String(bytes(name), Charsets.UTF_8)
    fun array(name: String) = JSONArray(text(name))
    fun obj(name: String) = JSONObject(text(name))

    fun JSONObject.optStr(k: String): String? = if (isNull(k)) null else getString(k)
    fun JSONArray.objects(): List<JSONObject> = (0 until length()).map { getJSONObject(it) }
    fun JSONArray.doubles(): DoubleArray = DoubleArray(length()) { getDouble(it) }
    fun JSONArray.strings(): List<String> = (0 until length()).map { getString(it) }
    fun JSONArray.nullableStrings(): List<String?> = (0 until length()).map { if (isNull(it)) null else getString(it) }

    fun sha256(b: ByteArray): String = java.security.MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    fun floatsLE(f: FloatArray): ByteArray {
        val bb = java.nio.ByteBuffer.allocate(f.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN); f.forEach { bb.putFloat(it) }; return bb.array()
    }

    fun floatsFromB64(s: String): FloatArray {
        val b = java.nio.ByteBuffer.wrap(java.util.Base64.getDecoder().decode(s)).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.remaining() / 4) { b.float }
    }

    @Suppress("unused") fun file(name: String) = File(Goldens::class.java.classLoader!!.getResource("goldens/$name")!!.toURI())
}
