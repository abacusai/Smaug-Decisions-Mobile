package ai.abacus.decide

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Android wiring of the cascade: the fast tier memory-mapped from the APK's uncompressed assets (assets/fast), Smaug-Decisions from the
 * app's external files dir (pushed by run_device_test.sh, or downloaded), the device tier from ActivityManager.MemoryInfo.
 */
object AndroidCascade {
    /** Smaug-Decisions backend of this build: "litert" (default) or "llamacpp" (-PbigBackend=llamacpp) */
    val BACKEND: String = BuildConfig.BIG_BACKEND
    const val LITERT_MODEL = "smaug-decisions-full-vis560.litertlm"
    const val GGUF = "smaug-decisions-Q4_K_M-imat.gguf"
    const val MMPROJ = "mmproj-smaug-decisions-q8_0.gguf"
    const val REPORT = "cascade-report.json"

    fun memoryInfo(ctx: Context): ActivityManager.MemoryInfo {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    }

    fun deviceTier(ctx: Context): DeviceTier = DevicePolicy.tier(memoryInfo(ctx).totalMem)

    /** an uncompressed asset as a read-only mapped buffer (no copy to storage, pages shared with the APK file). If the asset is
     *  stored compressed (openFd fails, e.g. an asset pack built without noCompress), it is extracted once to filesDir. */
    fun mapAsset(ctx: Context, name: String): ByteBuffer = try {
        ctx.assets.openFd(name).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.use { it.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length) }
        }
    } catch (e: java.io.FileNotFoundException) {
        val f = File(ctx.filesDir, name)
        if (!f.exists()) { f.parentFile!!.mkdirs(); val tmp = File(f.path + ".part"); ctx.assets.open(name).use { i -> tmp.outputStream().use { i.copyTo(it) } }; tmp.renameTo(f) }
        java.io.RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
    }

    fun fastTier(ctx: Context, threads: Int = AndroidSelfTest.defaultThreads()): FastTier = FastTier(FastTier.Source.Buffers(
        mapAsset(ctx, "fast/audio.onnx"), mapAsset(ctx, "fast/trunk.onnx"), embedTable(ctx),
        ctx.assets.open("fast/head.json").use { String(it.readBytes(), Charsets.UTF_8) }), threads)

    /** the shipped int8 table + scales; the fp16 table for asset sets built before it */
    fun embedTable(ctx: Context): EmbedTable =
        if (ctx.assets.list("fast")!!.contains(EmbedTable.INT8)) EmbedTable.int8(mapAsset(ctx, "fast/${EmbedTable.INT8}"), mapAsset(ctx, "fast/${EmbedTable.SCALES}"))
        else EmbedTable.fp16(mapAsset(ctx, "fast/${EmbedTable.FP16}"))

    fun scenarios(ctx: Context): List<Scenario> = Scenario.parseFile(ctx.assets.open("fast/scenarios.json").use { String(it.readBytes(), Charsets.UTF_8) })

    /** the Smaug-Decisions file(s) to load: the active variant from the models setup (LiteRT), or the GGUF pair (llama.cpp flavor) */
    fun bigFiles(ctx: Context, manager: ModelManager? = null): List<File> = ctx.getExternalFilesDir(null)!!.let { d ->
        if (BACKEND == "litert") listOf(File(d, (manager ?: ModelManager(ctx)).activeVariant?.file ?: LITERT_MODEL))
        else listOf(File(d, GGUF), File(d, MMPROJ))
    }

    fun bigInstalled(ctx: Context, manager: ModelManager? = null): Boolean = bigFiles(ctx, manager).all { it.exists() }

    /** LiteRT-LM's XNNPack weight cache (~2.8 GB, written on the first load): app-private, persistent, never cleared by the app.
     *  Without it Smaug-Decisions holds ~4.8 GB dirty instead of ~0.8 GB. */
    fun litertCacheDir(ctx: Context): File = File(ctx.filesDir, "litert_cache")

    /** Smaug-Decisions factory for this build's backend. llama.cpp: weights stay in the mmapped file (DEC_NO_REPACK=1). */
    /** Called around the one-time first load of a variant (LiteRT-LM writes its weight cache): start + expected seconds, then done. */
    interface PrepareListener { fun preparing(startMs: Long, expectedS: Double); fun prepared(seconds: Double) }

    fun bigFactory(ctx: Context, threads: Int = AndroidSelfTest.defaultThreads(), tier: DeviceTier = deviceTier(ctx),
                   manager: ModelManager? = null, listener: PrepareListener? = null): () -> BigModel = {
        val files = bigFiles(ctx, manager)
        require(files.all { it.exists() }) { "Smaug-Decisions not installed: push ${files.joinToString { it.name }} to ${files[0].parent}" }
        if (BACKEND == "litert") {
            val v = manager?.activeVariant
            val prepared = v == null || manager.isPrepared(v)
            if (!prepared) listener?.preparing(System.currentTimeMillis(), manager!!.expectedPrepareSeconds(v!!))
            val t0 = System.nanoTime()
            val make = { b: String -> LiteRtDecider(files[0].path, litertCacheDir(ctx), ctx.cacheDir, threads = threads, keepVision = DevicePolicy.keepVision(tier),
                visualBudget = v?.visionTokens ?: Prompt.IMAGE_TOKENS, vision = if (v == null || v.hasImage) LiteRtDecider.Vision.ON_DEMAND else LiteRtDecider.Vision.NEVER,
                audio = v?.hasAudio ?: true, engineBackend = b) }
            var backend = v?.let { manager.backendFor(it) } ?: ModelManager.BACKEND
            // As on iOS: a build the GPU delegate cannot take ("Failed to modify graph with delegate") loads on the CPU, and that
            // is remembered for the file.
            val d = try { make(backend) } catch (e: Exception) {
                if (backend != "gpu") throw e
                backend = "cpu"; v?.let { manager.rememberBackend(it, "cpu") }; make("cpu")
            }
            if (!prepared) {
                val s = (System.nanoTime() - t0) / 1e9
                ModelPrep.markPrepared(manager!!.prepStore, v!!.file, v.bytes, backend, s); listener?.prepared(s)
            }
            d
        } else { android.system.Os.setenv("DEC_NO_REPACK", "1", true); Decider(files[0].path, files[1].path, Decider.DEFAULT_B, threads = threads) }
    }

    fun cascade(ctx: Context, fast: FastTier, tau: Double = Cascade.DEFAULT_TAU, tier: DeviceTier = deviceTier(ctx),
                manager: ModelManager = ModelManager(ctx), listener: PrepareListener? = null): Cascade {
        return Cascade(fast, bigFactory(ctx, tier = tier, manager = manager, listener = listener), tau, tier) { kind, loaded ->
            (if (BACKEND == "litert") manager.fileBlocker(kind) else if (!bigInstalled(ctx, manager)) "Smaug-Decisions not installed" else null)
                ?: if (!DevicePolicy.canUseBig(tier, kind, memoryInfo(ctx).availMem, loaded)) "not enough free memory for Smaug-Decisions now" else null
        }.also { it.choice = manager.choice }
    }

    /** the cascade self-test on the device (assets/cascade_selftest); writes cascade-report.json next to the models */
    fun selfTest(ctx: Context, cascade: Cascade, log: (String) -> Unit = {}): JSONObject {
        val bundle = JSONObject(ctx.assets.open("cascade_selftest/expected.json").use { String(it.readBytes(), Charsets.UTF_8) })
        val r = CascadeSelfTest.run(cascade, bundle, scenarios(ctx).associateBy { it.id },
            { n -> ctx.assets.open("cascade_selftest/$n").use { it.readBytes() } }, AndroidSelfTest::decodeBitmap, log,
            if (BACKEND == "litert") "p_big_litert" else "p_big")
        val mi = memoryInfo(ctx)
        r.put("device", JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT)
            .put("abi", Build.SUPPORTED_ABIS.firstOrNull()).put("cores", Runtime.getRuntime().availableProcessors())
            .put("total_mem_mb", mi.totalMem / (1024 * 1024)).put("avail_mem_mb", mi.availMem / (1024 * 1024)).put("tier", cascade.deviceTier.name))
        File(ctx.getExternalFilesDir(null), REPORT).writeText(r.toString(1))
        return r
    }
}
