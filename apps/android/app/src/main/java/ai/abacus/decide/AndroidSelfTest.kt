package ai.abacus.decide

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import org.json.JSONObject
import java.io.File

/**
 * The self-test on a device: the bundle from assets/selftest, the GGUF + mmproj named in it from [modelDir] (default: the app's
 * external files dir, where run_device_test.sh pushes them). Writes selftest-report.json next to the models and returns it.
 */
object AndroidSelfTest {
    const val REPORT = "selftest-report.json"

    fun decodeBitmap(bytes: ByteArray): HfImage.Rgb {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888; inPremultiplied = false; inScaled = false }
        val bmp = requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)) { "BitmapFactory cannot decode" }
        val w = bmp.width; val h = bmp.height; val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h); bmp.recycle()
        val rgb = ByteArray(w * h * 3)
        for (i in px.indices) { val c = px[i]; rgb[3 * i] = (c shr 16).toByte(); rgb[3 * i + 1] = (c shr 8).toByte(); rgb[3 * i + 2] = c.toByte() }
        return HfImage.Rgb(rgb, w, h)
    }

    fun run(context: Context, modelDir: File = context.getExternalFilesDir(null)!!, threads: Int = defaultThreads(), repeat: Int = 1,
            only: Set<String>? = null, log: (String) -> Unit = {}): JSONObject {
        val assets = context.assets
        val litert = AndroidCascade.BACKEND == "litert"
        val dir = if (litert) "selftest_litert" else "selftest"
        val bundle = SelfTest.parse(assets.open("$dir/expected.json").use { String(it.readBytes(), Charsets.UTF_8) })
        val model = File(modelDir, bundle.modelFile)
        require(model.exists()) { "missing ${model.path} (adb push it, see run_device_test.sh)" }
        val (hwm0, _) = SelfTest.memoryMb()
        val t0 = System.nanoTime()
        val big: BigModel = if (litert) LiteRtDecider(model.path, AndroidCascade.litertCacheDir(context), context.cacheDir, threads = threads,
                                                      b = bundle.b)
                            else File(modelDir, bundle.mmprojFile).let { mm -> require(mm.exists()) { "missing ${mm.path}" }; Decider(model.path, mm.path, bundle.b, threads = threads) }
        val report = big.use { d ->
            val loadS = (System.nanoTime() - t0) / 1e9
            val (_, rssLoaded) = SelfTest.memoryMb()
            log("loaded in %.1f s, %d threads, RSS %.0f MB".format(loadS, threads, rssLoaded))
            SelfTest.run(d, bundle, { name -> assets.open("$dir/$name").use { it.readBytes() } }, ::decodeBitmap, repeat, only, log).also {
                it.getJSONObject("summary").put("load_s", loadS).put("threads", threads).put("rss_after_load_mb", rssLoaded)
                    .put("peak_rss_before_load_mb", hwm0).put("backend", d.backend).put("smaps_end_mb", SelfTest.smapsMb())
            }
        }
        report.put("device", JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("soc", socName())
            .put("sdk", Build.VERSION.SDK_INT).put("abi", Build.SUPPORTED_ABIS.firstOrNull()).put("cores", Runtime.getRuntime().availableProcessors())
            .put("ram_mb", ramMb(context)))
        File(modelDir, REPORT).writeText(report.toString(1))
        return report
    }

    /** big + mid cores on typical phones; llama.cpp gets slower when it spills onto little cores */
    fun defaultThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

    private fun socName(): String = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE

    private fun ramMb(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        return android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem / (1024 * 1024)
    }
}
