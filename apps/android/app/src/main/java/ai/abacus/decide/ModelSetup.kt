package ai.abacus.decide

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** What the user chose to install and use (Models screen, and the first-run "Get Smaug-Decisions" card). Mirrors the iOS ModelChoice. */
enum class ModelChoice(val id: String, val title: String, val detail: String, val bigModalities: Set<String>) {
    AUDIO_ONLY("audio_only", "0.7B audio model only", "Audio only · already on this phone", emptySet()),
    TEXT_IMAGE("text_image", "Smaug-Decisions for text + images", "Text + photos · audio: 0.7B audio model", setOf("text", "image")),
    AUDIO_TEXT("audio_text", "Smaug-Decisions for audio + text", "Unsure audio + text · no photos", setOf("audio", "text")),
    FULL("full", "Full cascade", "Audio, text and photos", setOf("audio", "text", "image"));

    val usesBig get() = bigModalities.isNotEmpty()
    val escalatesAudio get() = "audio" in bigModalities
    /** The audio model always answers audio; text and images need Smaug-Decisions with that modality. */
    fun answers(kind: String) = kind == "audio" || kind in bigModalities

    companion object {
        fun offered(tier: DeviceTier): List<ModelChoice> = if (tier == DeviceTier.FAST_ONLY) listOf(AUDIO_ONLY) else entries
        fun default(tier: DeviceTier) = if (tier == DeviceTier.FAST_ONLY) AUDIO_ONLY else FULL
        fun of(id: String?) = entries.firstOrNull { it.id == id }
        /** the stored choice, limited to what the device allows */
        fun current(tier: DeviceTier, stored: String?): ModelChoice = of(stored)?.takeIf { it in offered(tier) } ?: default(tier)
        /** image soft tokens of the build for a tier: vis560 on 8 GB+, vis280 on 6 GB phones, none on fast-only */
        fun visionTokens(tier: DeviceTier): Int? = when (tier) { DeviceTier.FULL, DeviceTier.STANDARD -> 560; DeviceTier.SQUEEZE -> 280; else -> null }
    }
}

/** The first-run "Choose your setup" screen: shown once, before the home screen, while no Smaug-Decisions file is on the device. */
object FirstRun {
    const val KEY_SEEN = "firstrun.setup_seen"

    fun shouldShow(seen: Boolean, bigOnDevice: Boolean, downloadPending: Boolean) = !seen && !bigOnDevice && !downloadPending

    /** the option marked "Recommended" and preselected: the device's default choice */
    fun recommended(tier: DeviceTier) = ModelChoice.default(tier)
}

/**
 * Where Smaug-Decisions' files come from: a small JSON manifest. Each variant has exactly the quant build's fields (variant, file,
 * bytes, sha256, modalities, vision_tokens); around them `base_url` (download = base_url + file), optional `url_overrides`
 * (file -> URL, e.g. a test file) and `estimated` (variants whose size / sha256 are placeholders; not downloadable).
 * Bundled as assets/models_manifest.json, overridden by <external files dir>/models_manifest.json.
 */
class ModelManifest(val baseUrl: String, val urlOverrides: Map<String, String>, val estimated: Set<String>, val variants: List<Variant>,
                      val source: String) {
    data class Variant(val variant: String, val file: String, val bytes: Long, val sha256: String, val modalities: Set<String>, val visionTokens: Int?,
                       /** the published manifest's per-file download URL (else base_url + file) */
                       val url: String? = null) {
        val hasAudio get() = "audio" in modalities
        val hasImage get() = "image" in modalities
        /** The one-time LiteRT-LM weight cache, estimated from the size (same formula as iOS): decoder cache ~0.69 x the
         *  decoder's share of the file (measured 2.04 GB for every Smaug-Decisions-LiteRT variant) + 0.30 GB audio + 0.16 GB vision encoder. */
        val cacheBytes: Long get() {
            val decoderFile = bytes - (if (hasAudio) 300_000_000L else 0) - (if (hasImage) 170_000_000L else 0)
            return (decoderFile * 0.69).toLong() + (if (hasAudio) 300_000_000L else 0) + (if (hasImage) 160_000_000L else 0)
        }
        val cacheBytesWithoutVision: Long get() = cacheBytes - (if (hasImage) 160_000_000L else 0)
    }

    fun isEstimated(v: Variant) = v.variant in estimated
    fun url(v: Variant): String = urlOverrides[v.file] ?: v.url ?: (baseUrl + v.file)

    /** smallest variant with every modality the choice needs, at the device's image size when the choice includes images */
    fun variantFor(choice: ModelChoice, visionTokens: Int?): Variant? {
        val need = choice.bigModalities
        if (need.isEmpty()) return null
        return variants.filter { v -> v.modalities.containsAll(need) && ("image" !in need || visionTokens == null || v.visionTokens == visionTokens) }
            .minWithOrNull(compareBy({ it.bytes }, { it.variant }))
    }

    companion object {
        fun parse(json: String, source: String): ModelManifest {
            val o = JSONObject(json)
            val ov = o.optJSONObject("url_overrides")?.let { m -> m.keys().asSequence().associateWith { m.getString(it) } } ?: emptyMap()
            val est = o.optJSONArray("estimated")?.let { a -> (0 until a.length()).map { a.getString(it) }.toSet() } ?: emptySet()
            val vs = o.getJSONArray("variants").let { a ->
                (0 until a.length()).map { i ->
                    val v = a.getJSONObject(i); val m = v.getJSONArray("modalities")
                    Variant(v.getString("variant"), v.getString("file"), v.getLong("bytes"), v.getString("sha256"),
                        (0 until m.length()).map { m.getString(it) }.toSet(), if (v.isNull("vision_tokens")) null else v.getInt("vision_tokens"),
                        v.optString("url").ifEmpty { null })
                }
            }
            return ModelManifest(o.getString("base_url"), ov, est, vs, source)
        }

        /** the override in [dir] (models_manifest.json) if it parses, else the bundled one */
        fun load(dir: File?, bundled: () -> String?): ModelManifest? {
            dir?.let { File(it, FILE) }?.takeIf { it.exists() }?.let { f -> runCatching { return parse(f.readText(), "override") } }
            return bundled()?.let { runCatching { parse(it, "bundled") }.getOrNull() }
        }

        const val FILE = "models_manifest.json"
    }
}

/** What switching to a choice does to the files: download the target if missing, offer to delete the other variants. */
data class ModelPlan(val choice: ModelChoice, val target: ModelManifest.Variant?, val download: ModelManifest.Variant?,
                     val delete: List<ModelManifest.Variant>) {
    val freedBytes get() = delete.sumOf { it.bytes }

    companion object {
        fun make(choice: ModelChoice, manifest: ModelManifest?, visionTokens: Int?, installed: (String) -> Boolean): ModelPlan {
            val target = manifest?.variantFor(choice, visionTokens)
            val others = (manifest?.variants ?: emptyList()).filter { it.file != target?.file && installed(it.file) }.distinctBy { it.file }
            return ModelPlan(choice, target, target?.takeIf { !installed(it.file) }, others)
        }
    }
}

/** Free storage and what Smaug-Decisions needs of it. */
object ModelStorage {
    const val HEADROOM = 500_000_000L

    /** the rest of the file, its whole weight cache, and headroom */
    fun downloadNeed(v: ModelManifest.Variant, alreadyDownloaded: Long = 0) = maxOf(0L, v.bytes - alreadyDownloaded) + v.cacheBytes + HEADROOM

    /** the part of the weight cache not written yet, and headroom */
    fun firstLoadNeed(v: ModelManifest.Variant, existingCache: Long, vision: Boolean) =
        maxOf(0L, (if (vision) v.cacheBytes else v.cacheBytesWithoutVision) - existingCache) + HEADROOM

    fun gb(b: Long) = "%.1f GB".format(b / 1e9)
    fun gb2(b: Long) = "%.2f GB".format(b / 1e9)

    /** "<what> needs 7.9 GB free and 2.1 GB is free. Free up 5.8 GB ...", or null when it fits */
    fun shortfall(need: Long, free: Long, what: String): String? =
        if (free >= need) null else "$what needs ${gb(need)} free and ${gb(free)} is free. Free up ${gb(need - free)} in Settings > Storage, then try again."

    /** bytes of LiteRT-LM cache files written for one model file (named "<file>_<mtime>_<size>.<part>.xnnpack_cache") */
    fun cacheBytes(cacheDir: File, file: String): Long = cacheDir.listFiles()?.filter { it.name.startsWith(file + "_") }?.sumOf { it.length() } ?: 0L

    /** deletes cache files of model files that are not in [present]; returns the bytes freed */
    /** Deletes the caches of model files that are gone, and of older copies of present ones: LiteRT-LM names a cache
     *  "<file>_<mtime s>_<size>.<part>", so a replaced file (new mtime or size) leaves a stale cache. [modelDir] holds the files. */
    fun dropStaleCaches(cacheDir: File, present: Set<String>, modelDir: File? = null): Long {
        var freed = 0L
        cacheDir.listFiles()?.forEach { f ->
            val i = f.name.indexOf(".litertlm_"); if (i < 0) return@forEach
            val model = f.name.substring(0, i) + ".litertlm"
            val token = modelDir?.let { File(it, model) }?.takeIf { it.exists() }?.let { "${it.lastModified() / 1000}_${it.length()}" }
            val rest = f.name.substring(i + ".litertlm_".length)   // "<mtime>_<size>.<part>" (CPU) or "<mtime>_<size>_mldrift_*" (GPU)
            val current = model in present && (token == null || rest.startsWith("$token.") || rest.startsWith("${token}_"))
            if (!current) { freed += f.length(); f.delete() }
        }
        return freed
    }
}

/**
 * The one-time "Preparing Smaug-Decisions" step (LiteRT-LM writes its weight cache on the first load of a file). Done is stored per
 * file, size and backend (the store is SharedPreferences on Android, a map in tests) and only counts while the file's cache is
 * still on disk.
 */
object ModelPrep {
    interface Store { fun getBool(k: String): Boolean; fun putBool(k: String, v: Boolean); fun getDouble(k: String): Double; fun putDouble(k: String, v: Double) }

    fun key(file: String, bytes: Long, backend: String) = "smaug.prepared.$file.$bytes.$backend"

    fun isPrepared(s: Store, file: String, bytes: Long, backend: String, cacheBytes: Long): Boolean =
        cacheBytes >= minOf(1_000_000_000L, bytes / 4) && s.getBool(key(file, bytes, backend))

    fun markPrepared(s: Store, file: String, bytes: Long, backend: String, seconds: Double) {
        s.putBool(key(file, bytes, backend), true)
        s.putDouble("smaug.prepare_s_per_3_43gb.$backend", seconds * 3.43e9 / maxOf(bytes, 1L))
    }

    /** the last prepare time measured on this device, scaled to the file; else 45 s per 3.43 GB (unmeasured on Android phones) */
    fun expectedSeconds(s: Store, bytes: Long, backend: String): Double {
        val m = s.getDouble("smaug.prepare_s_per_3_43gb.$backend")
        return maxOf(5.0, (if (m > 3) m else 45.0) * bytes / 3.43e9)
    }
}

/**
 * A resumable HTTP download into `<file>.part` next to the destination: a Range request continues a partial file (200 instead
 * of 206 restarts it), redirects are followed, then size + sha256 are checked and the file is renamed into place. A mismatch
 * deletes the partial file. [cancel] stops it at the next buffer and keeps the partial file for a later resume.
 */
class ResumableDownloader(val item: Item, val dir: File, private val open: (URL) -> HttpURLConnection = { it.openConnection() as HttpURLConnection }) {
    data class Item(val file: String, val url: String, val sha256: String, val bytes: Long)

    sealed class Result {
        data class Done(val file: File) : Result()
        data class Cancelled(val received: Long) : Result()
        data class Failed(val message: String, val retryable: Boolean) : Result()
    }

    val part = File(dir, item.file + ".part")
    val dest = File(dir, item.file)
    @Volatile var cancelled = false; private set
    /** the Range offset of the last request (0 = from the start), for tests and logs */
    var lastRangeStart = -1L; private set

    fun cancel() { cancelled = true }

    fun run(progress: (received: Long, total: Long) -> Unit = { _, _ -> }, verifying: (Double) -> Unit = {}): Result {
        dir.mkdirs()
        var have = if (part.exists()) part.length() else 0L
        if (have > item.bytes) { part.delete(); have = 0 }
        if (have < item.bytes) {
            val c = try { open(URL(item.url)) } catch (e: Exception) { return Result.Failed("Could not reach the download server.", true) }
            try {
                c.instanceFollowRedirects = true; c.connectTimeout = 30_000; c.readTimeout = 60_000
                if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
                lastRangeStart = have
                val code = c.responseCode
                when {
                    code == 401 || code == 403 -> return Result.Failed("The download server refused (HTTP $code). Copy the file in with adb instead.", false)
                    code == 416 && have == item.bytes -> {}
                    code == 200 -> { have = 0; part.delete() }   // no range support (or a full response): start over
                    code == 206 -> {}
                    else -> return Result.Failed("The download failed (HTTP $code).", code >= 500)
                }
                if (code == 200 || code == 206) RandomAccessFile(part, "rw").use { out ->
                    out.seek(have)
                    c.inputStream.use { inp ->
                        val buf = ByteArray(1 shl 20); var last = 0L
                        while (true) {
                            if (cancelled) return Result.Cancelled(have)
                            val n = inp.read(buf); if (n < 0) break
                            out.write(buf, 0, n); have += n
                            if (have - last >= (4 shl 20) || have == item.bytes) { last = have; progress(have, item.bytes) }
                        }
                    }
                }
            } catch (e: java.io.IOException) {
                return if (cancelled) Result.Cancelled(part.length()) else Result.Failed("Download interrupted (${e.javaClass.simpleName}). It resumes where it stopped.", true)
            } finally { c.disconnect() }
        }
        val n = part.length()
        if (n != item.bytes) return if (n < item.bytes) Result.Failed("Download interrupted at ${ModelStorage.gb2(n)}. It resumes where it stopped.", true)
                                    else { part.delete(); Result.Failed("The download has $n bytes, the manifest says ${item.bytes}. It was deleted; try again.", true) }
        val hex = sha256(part, verifying)
        if (hex != item.sha256.lowercase()) { part.delete(); return Result.Failed("Checksum mismatch: the download was corrupted. It was deleted; try again.", true) }
        if (dest.exists()) dest.delete()
        if (!part.renameTo(dest)) return Result.Failed("Could not move the download into place.", true)
        return Result.Done(dest)
    }

    companion object {
        fun sha256(f: File, progress: (Double) -> Unit = {}): String {
            val md = MessageDigest.getInstance("SHA-256"); val total = maxOf(1L, f.length()); var done = 0L; var last = 0L
            f.inputStream().use { inp ->
                val buf = ByteArray(8 shl 20)
                while (true) { val n = inp.read(buf); if (n < 0) break; md.update(buf, 0, n); done += n; if (done - last > 100_000_000) { last = done; progress(done.toDouble() / total) } }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
