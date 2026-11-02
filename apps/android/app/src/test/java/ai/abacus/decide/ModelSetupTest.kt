package ai.abacus.decide

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.net.ServerSocket
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.random.Random

/** Models setup: routing per choice, offered choices, the real manifest, switch plans, storage math, prepare, the downloader. */
class ModelSetupTest {
    private val assets = File(System.getProperty("decide.assets") ?: "src/main/assets")
    private fun manifest() = ModelManifest.parse(File(assets, "models_manifest.json").readText(), "bundled")

    @Test fun routingPerChoice() {
        val R = Cascade.Route.entries.associateBy { it.name }
        // choice -> audio confident, audio unsure, audio unsure without Smaug-Decisions now, text, image, text without Smaug-Decisions now
        val table = mapOf(
            ModelChoice.FULL to listOf("FAST", "ESCALATED", "FLAGGED", "ROUTED", "ROUTED", "FLAGGED"),
            ModelChoice.AUDIO_TEXT to listOf("FAST", "ESCALATED", "FLAGGED", "ROUTED", "NOT_INSTALLED", "FLAGGED"),
            ModelChoice.TEXT_IMAGE to listOf("FAST", "UNSURE", "UNSURE", "ROUTED", "ROUTED", "FLAGGED"),
            ModelChoice.AUDIO_ONLY to listOf("FAST", "UNSURE", "UNSURE", "NOT_INSTALLED", "NOT_INSTALLED", "NOT_INSTALLED"))
        for ((c, e) in table) {
            val got = listOf(Cascade.route("audio", 0.95, 0.9, true, c), Cascade.route("audio", 0.6, 0.9, true, c), Cascade.route("audio", 0.6, 0.9, false, c),
                Cascade.route("text", null, 0.9, true, c), Cascade.route("image", null, 0.9, true, c), Cascade.route("text", null, 0.9, false, c))
            assertEquals(c.name, e.map { R.getValue(it) }, got)
        }
        assertEquals(Cascade.Route.FAST, Cascade.route("audio", 0.9, 0.9, true))   // >= tau answers on the fast tier
        assertEquals(Cascade.Route.ESCALATED, Cascade.route("audio", null, 0.9, true))
    }

    @Test fun offeredChoicesFollowTheDeviceTier() {
        assertEquals(listOf(ModelChoice.AUDIO_ONLY), ModelChoice.offered(DeviceTier.FAST_ONLY))
        assertEquals(ModelChoice.entries, ModelChoice.offered(DeviceTier.SQUEEZE))
        assertEquals(ModelChoice.FULL, ModelChoice.current(DeviceTier.STANDARD, null))
        assertEquals(ModelChoice.TEXT_IMAGE, ModelChoice.current(DeviceTier.STANDARD, "text_image"))
        assertEquals(ModelChoice.AUDIO_ONLY, ModelChoice.current(DeviceTier.FAST_ONLY, "full"))   // never Smaug-Decisions on fast-only
        assertEquals(560, ModelChoice.visionTokens(DeviceTier.STANDARD)); assertEquals(280, ModelChoice.visionTokens(DeviceTier.SQUEEZE))
    }

    @Test fun firstRunShowsOnceBeforeAnyBigFile() {
        assertTrue(FirstRun.shouldShow(seen = false, bigOnDevice = false, downloadPending = false))
        assertFalse(FirstRun.shouldShow(seen = true, bigOnDevice = false, downloadPending = false))
        assertFalse(FirstRun.shouldShow(seen = false, bigOnDevice = true, downloadPending = false))
        assertFalse(FirstRun.shouldShow(seen = false, bigOnDevice = false, downloadPending = true))
        for (t in DeviceTier.entries) assertTrue(t.name, FirstRun.recommended(t) in ModelChoice.offered(t))
        assertEquals(ModelChoice.FULL, FirstRun.recommended(DeviceTier.SQUEEZE))
        assertEquals(ModelChoice.AUDIO_ONLY, FirstRun.recommended(DeviceTier.FAST_ONLY))
        for (c in ModelChoice.entries) assertTrue(c.name, c.detail.isNotEmpty() && c.detail.length <= 40)   // one line at 360 dp
    }

    @Test fun realManifestResolvesEveryChoice() {
        val m = manifest()
        assertEquals(6, m.variants.size)
        for (v in m.variants) { assertEquals(64, v.sha256.length); assertFalse(m.isEstimated(v)) }
        assertNull(m.variantFor(ModelChoice.AUDIO_ONLY, 560))
        assertEquals("full_vis560", m.variantFor(ModelChoice.FULL, 560)!!.variant)
        assertEquals("full_vis280", m.variantFor(ModelChoice.FULL, 280)!!.variant)
        assertEquals("noaudio_vis280", m.variantFor(ModelChoice.TEXT_IMAGE, 280)!!.variant)
        assertEquals("novision", m.variantFor(ModelChoice.AUDIO_TEXT, 560)!!.variant)
        assertNull(m.variants.first { it.variant == "novision" }.visionTokens)
        assertEquals("https://huggingface.co/abacusai/Smaug-Decisions-LiteRT/resolve/main/smaug-decisions-novision.litertlm", m.url(m.variantFor(ModelChoice.AUDIO_TEXT, 560)!!))
        // decoder cache estimate vs the measured 2,044,028,752 B for every Smaug-Decisions-LiteRT variant
        for (v in m.variants) assertEquals(v.variant, 2.044e9, (v.cacheBytesWithoutVision - if (v.hasAudio) 300_000_000L else 0).toDouble(), 0.05e9)
    }

    @Test fun switchPlanAndStorage() {
        val m = manifest(); val have = setOf("smaug-decisions-full-vis280.litertlm")
        val keep = ModelPlan.make(ModelChoice.FULL, m, 280, have::contains)
        assertNull(keep.download); assertTrue(keep.delete.isEmpty())
        val smaller = ModelPlan.make(ModelChoice.AUDIO_TEXT, m, 280, have::contains)
        assertEquals("novision", smaller.download!!.variant); assertEquals(listOf("smaug-decisions-full-vis280.litertlm"), smaller.delete.map { it.file })
        val audioOnly = ModelPlan.make(ModelChoice.AUDIO_ONLY, m, 280, have::contains)
        assertNull(audioOnly.target); assertEquals(3_449_280_464L, audioOnly.freedBytes)
        val full = m.variantFor(ModelChoice.FULL, 280)!!
        assertEquals(full.bytes + full.cacheBytes + ModelStorage.HEADROOM, ModelStorage.downloadNeed(full))
        assertEquals(full.cacheBytes + ModelStorage.HEADROOM, ModelStorage.downloadNeed(full, full.bytes))
        assertNull(ModelStorage.shortfall(7_000_000_000, 8_000_000_000, "Smaug-Decisions"))
        assertTrue(ModelStorage.shortfall(7_000_000_000, 2_000_000_000, "Smaug-Decisions")!!.contains("Free up 5.0 GB"))
        assertEquals(ModelStorage.HEADROOM, ModelStorage.firstLoadNeed(full, full.cacheBytesWithoutVision, vision = false))
    }

    /** Caches are kept for the current copy of a present file (CPU ".xnnpack_cache" and GPU "_mldrift_*" names) and dropped
     *  for a replaced copy (other mtime/size) or a missing file. */
    @Test fun staleCachesFollowTheFileToken() {
        val models = kotlin.io.path.createTempDirectory("models").toFile(); val cache = kotlin.io.path.createTempDirectory("cache").toFile()
        val f = File(models, "smaug-decisions-full-vis280.litertlm").apply { writeBytes(ByteArray(10)); setLastModified(1_793_378_187_000L) }
        val token = "${f.lastModified() / 1000}_${f.length()}"
        val keep = listOf("${f.name}_$token.xnnpack_cache", "${f.name}_$token.vision_encoder.xnnpack_cache", "${f.name}_${token}_mldrift_weight_cache.bin")
        val drop = listOf("${f.name}_1700000000_10.xnnpack_cache", "${f.name}_${token}9_mldrift_program_cache.bin", "gone.litertlm_1_2.xnnpack_cache")
        (keep + drop).forEach { File(cache, it).writeBytes(ByteArray(3)) }
        val freed = ModelStorage.dropStaleCaches(cache, setOf(f.name), models)
        assertEquals(keep.toSet(), cache.list()!!.toSet())
        assertEquals(9L, freed)
    }

    @Test fun prepareIsPersistedAndNeedsTheCache() {
        val mem = HashMap<String, Any>()
        val s = object : ModelPrep.Store {
            override fun getBool(k: String) = mem[k] as? Boolean ?: false
            override fun putBool(k: String, v: Boolean) { mem[k] = v }
            override fun getDouble(k: String) = mem[k] as? Double ?: 0.0
            override fun putDouble(k: String, v: Double) { mem[k] = v }
        }
        val f = "smaug-decisions-full-vis560.litertlm"; val b = 3_428_571_088L
        assertEquals(45.0, ModelPrep.expectedSeconds(s, b, "cpu"), 0.1)
        assertFalse(ModelPrep.isPrepared(s, f, b, "cpu", 2_000_000_000))
        ModelPrep.markPrepared(s, f, b, "cpu", 30.0)
        assertTrue(ModelPrep.isPrepared(s, f, b, "cpu", 2_000_000_000))
        assertFalse("cache deleted -> prepare again", ModelPrep.isPrepared(s, f, b, "cpu", 0))
        assertEquals(30.0, ModelPrep.expectedSeconds(s, b, "cpu"), 0.1)
        val dir = Files.createTempDirectory("cache").toFile()
        File(dir, "$f" + "_1_2.xnnpack_cache").writeBytes(ByteArray(10)); File(dir, "old.litertlm_1_2.xnnpack_cache").writeBytes(ByteArray(7))
        assertEquals(10L, ModelStorage.cacheBytes(dir, f))
        assertEquals(7L, ModelStorage.dropStaleCaches(dir, setOf(f))); assertEquals(1, dir.listFiles()!!.size)
    }

    /** a local server with Range support; the first transfer is interrupted, the second resumes from the partial file */
    @Test fun downloaderResumesVerifiesAndRejectsBadChecksums() {
        val body = Random(7).nextBytes(9_000_000)
        val sha = MessageDigest.getInstance("SHA-256").digest(body).joinToString("") { "%02x".format(it) }
        val ranges = mutableListOf<String?>()
        // a minimal HTTP/1.1 file server with Range support (com.sun.net.httpserver is not on the unit-test classpath)
        val srv = ServerSocket(0)
        val server = Thread {
            while (!srv.isClosed) {
                val sock = runCatching { srv.accept() }.getOrNull() ?: break
                Thread {
                    sock.use { so ->
                        val inp = so.getInputStream().bufferedReader(); var range: String? = null
                        while (true) { val l = inp.readLine() ?: break; if (l.isEmpty()) break; if (l.startsWith("Range:", true)) range = l.substringAfter(':').trim() }
                        synchronized(ranges) { ranges += range }
                        val from = range?.removePrefix("bytes=")?.substringBefore('-')?.toInt() ?: 0
                        val head = (if (from > 0) "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes $from-${body.size - 1}/${body.size}\r\n" else "HTTP/1.1 200 OK\r\n") +
                            "Accept-Ranges: bytes\r\nContent-Length: ${body.size - from}\r\nConnection: close\r\n\r\n"
                        val out = so.getOutputStream()
                        runCatching { out.write(head.toByteArray()); var i = from; while (i < body.size) { val n = minOf(65536, body.size - i); out.write(body, i, n); i += n }; out.flush() }
                    }
                }.start()
            }
        }.apply { isDaemon = true; start() }
        try {
            val url = "http://127.0.0.1:${srv.localPort}/f.bin"
            val dir = Files.createTempDirectory("dl").toFile()
            val item = ResumableDownloader.Item("f.bin", url, sha, body.size.toLong())
            val d1 = ResumableDownloader(item, dir)
            val r1 = d1.run({ got, _ -> if (got >= 4_000_000) d1.cancel() })
            assertTrue("$r1", r1 is ResumableDownloader.Result.Cancelled)
            val partial = d1.part.length(); assertTrue(partial in 4_000_000 until body.size)
            val d2 = ResumableDownloader(item, dir)
            val r2 = d2.run()
            assertTrue("$r2", r2 is ResumableDownloader.Result.Done)
            assertEquals(partial, d2.lastRangeStart)
            assertEquals("bytes=$partial-", ranges.last())
            assertTrue(File(dir, "f.bin").readBytes().contentEquals(body)); assertFalse(d2.part.exists())
            // wrong checksum: verified, rejected, partial file deleted, nothing moved into place
            val bad = ResumableDownloader(item.copy(file = "g.bin", sha256 = "0".repeat(64)), dir)
            val r3 = bad.run()
            assertTrue("$r3", r3 is ResumableDownloader.Result.Failed && r3.message.contains("Checksum"))
            assertFalse(bad.part.exists()); assertFalse(bad.dest.exists())
        } finally { srv.close(); server.join(1000) }
    }

    /** -Pdecide.liveDownload=true: the public 438 MB test file through a fake manifest, interrupted at ~60 MB and resumed. */
    @Test fun liveDownloadThroughAFakeManifest() {
        assumeTrue("live download skipped: pass -Pdecide.liveDownload=true", System.getProperty("decide.liveDownload") == "true")
        val fake = File("../../ios/tools/testdata/models_manifest_dltest.json")
        val m = ModelManifest.parse(fake.readText(), "override")
        val v = m.variantFor(ModelChoice.FULL, 560); assertNotNull(v)
        val dir = Files.createTempDirectory("live").toFile()
        val item = ResumableDownloader.Item(v!!.file, m.url(v), v.sha256, v.bytes)
        val t0 = System.nanoTime()
        val d1 = ResumableDownloader(item, dir)
        val r1 = d1.run({ got, _ -> if (got >= 60_000_000) d1.cancel() })
        val partial = d1.part.length()
        val d2 = ResumableDownloader(item, dir)
        var verifyCalls = 0
        val r2 = d2.run(verifying = { verifyCalls++ })
        val s = (System.nanoTime() - t0) / 1e9
        println("LIVE DOWNLOAD first=$r1 partial=$partial resumedFrom=${d2.lastRangeStart} result=$r2 seconds=%.1f size=${File(dir, v.file).length()}".format(s))
        assertTrue(r1 is ResumableDownloader.Result.Cancelled); assertEquals(partial, d2.lastRangeStart)
        assertTrue("$r2", r2 is ResumableDownloader.Result.Done); assertEquals(v.bytes, File(dir, v.file).length())
        File(dir, v.file).delete()
    }
}
