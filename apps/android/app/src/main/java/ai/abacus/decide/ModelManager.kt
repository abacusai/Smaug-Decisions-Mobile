package ai.abacus.decide

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.Worker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import java.io.File

/**
 * Android side of the models setup: the manifest (bundled assets/models_manifest.json, overridable by
 * <external files dir>/models_manifest.json), the stored choice, which variant file is active, storage checks, the one-time
 * prepare bookkeeping, and the background download (WorkManager, [ModelDownloadWorker]).
 */
class ModelManager(private val ctx: Context) {
    val prefs: SharedPreferences = ctx.getSharedPreferences("cascade", Context.MODE_PRIVATE)
    val dir: File = ctx.getExternalFilesDir(null)!!
    val cacheDir: File = AndroidCascade.litertCacheDir(ctx)
    val tier: DeviceTier = AndroidCascade.deviceTier(ctx)
    val visionTokens: Int? = ModelChoice.visionTokens(tier)
    @Volatile var manifest: ModelManifest? = loadManifest(); private set
    /** the running download's progress (0..100), set from the UI's WorkInfo observer; -1 when none runs */
    @Volatile var downloadPercent = -1

    val prepStore = object : ModelPrep.Store {
        override fun getBool(k: String) = prefs.getBoolean(k, false)
        override fun putBool(k: String, v: Boolean) { prefs.edit().putBoolean(k, v).apply() }
        override fun getDouble(k: String) = prefs.getFloat(k, 0f).toDouble()
        override fun putDouble(k: String, v: Double) { prefs.edit().putFloat(k, v.toFloat()).apply() }
    }

    init {
        // the weight cache of a model file that is gone, or of an older copy of one, is 2-3 GB of dead weight
        val models = dir.listFiles()?.map { it.name }?.filter { it.endsWith(".litertlm") }?.toSet() ?: emptySet()
        ModelStorage.dropStaleCaches(cacheDir, models, dir)
    }

    /**
     * A build renamed in the manifest (an older file name -> smaug-decisions-full-vis280): an unlisted .litertlm in the files dir with a
     * listed variant's size and sha256 is renamed to it, with its weight cache (rename keeps the mtime that LiteRT-LM puts in
     * the cache names), remembered backend and prepared flag. Hashes ~3.4 GB, so call it off the main thread. Returns "old -> new".
     */
    fun migrateRenamed(): List<String> {
        val m = manifest ?: return emptyList()
        val listed = m.variants.map { it.file }.toSet()
        val done = mutableListOf<String>()
        for (f in dir.listFiles()?.filter { it.name.endsWith(".litertlm") && it.name !in listed } ?: emptyList()) {
            val size = f.length()
            val cands = m.variants.filter { it.bytes == size && !installed(it.file) }
            if (cands.isEmpty()) continue
            val sha = ResumableDownloader.sha256(f) {}
            val v = cands.firstOrNull { it.sha256.equals(sha, ignoreCase = true) } ?: continue
            val old = f.name
            if (!f.renameTo(File(dir, v.file))) continue
            cacheDir.listFiles()?.filter { it.name.startsWith(old + "_") }?.forEach { it.renameTo(File(cacheDir, v.file + it.name.substring(old.length))) }
            val e = prefs.edit()
            for (b in listOf("cpu", "gpu")) if (prepStore.getBool(ModelPrep.key(old, size, b))) e.putBoolean(ModelPrep.key(v.file, v.bytes, b), true)
            prefs.getString("smaug.backend.$old.$size", null)?.let { e.putString("smaug.backend.${v.file}.${v.bytes}", it) }
            e.apply()
            done += "$old -> ${v.file}"
        }
        return done
    }

    /** The decoder backend for a model file: what worked for it before (a failed GPU load falls back to the CPU and is
     *  remembered), else [BACKEND]. */
    fun backendFor(v: ModelManifest.Variant): String = prefs.getString("smaug.backend.${v.file}.${v.bytes}", null) ?: BACKEND
    fun rememberBackend(v: ModelManifest.Variant, b: String) { prefs.edit().putString("smaug.backend.${v.file}.${v.bytes}", b).apply() }

    fun loadManifest(): ModelManifest? = ModelManifest.load(dir) { runCatching { ctx.assets.open(ModelManifest.FILE).use { String(it.readBytes()) } }.getOrNull() }
    fun reloadManifest() { manifest = loadManifest() }

    val offered: List<ModelChoice> get() = ModelChoice.offered(tier)
    val choice: ModelChoice get() = ModelChoice.current(tier, prefs.getString(KEY_CHOICE, null))

    fun installed(file: String) = File(dir, file).exists()
    val installedFiles: Set<String> get() = (manifest?.variants ?: emptyList()).map { it.file }.filter(::installed).toSet()
    fun plan(c: ModelChoice) = ModelPlan.make(c, manifest, visionTokens, ::installed)

    /** the file the current choice uses: its own variant when installed, else an installed variant that covers it */
    val activeVariant: ModelManifest.Variant? get() {
        val c = choice; val m = manifest ?: return null
        if (!c.usesBig) return null
        plan(c).target?.takeIf { installed(it.file) }?.let { return it }
        return m.variants.filter { v -> v.modalities.containsAll(c.bigModalities) && installed(v.file) &&
            ("image" !in c.bigModalities || visionTokens == null || v.visionTokens == visionTokens) }.minByOrNull { it.bytes }
    }

    /** the variant the Get Smaug-Decisions card offers: Smaug-Decisions allowed here, the choice uses it, no usable file on the device */
    val pendingDownload: ModelManifest.Variant? get() =
        if (tier == DeviceTier.FAST_ONLY || !choice.usesBig || activeVariant != null) null else plan(choice).target

    fun freeBytes(): Long = dir.usableSpace

    /** At launch: not seen yet (a choice stored from the Models screen counts as seen), no .litertlm here (downloaded, pushed,
     *  or awaiting a rename) and no partial download. */
    fun firstRunPending(): Boolean {
        val names = dir.listFiles()?.map { it.name } ?: emptyList()
        return FirstRun.shouldShow(prefs.getBoolean(FirstRun.KEY_SEEN, false) || prefs.contains(KEY_CHOICE),
            names.any { it.endsWith(".litertlm") }, names.any { it.endsWith(".litertlm.part") })
    }
    fun markFirstRunSeen() { prefs.edit().putBoolean(FirstRun.KEY_SEEN, true).apply() }

    fun partBytes(v: ModelManifest.Variant) = File(dir, v.file + ".part").takeIf { it.exists() }?.length() ?: 0L

    fun downloadBlocker(v: ModelManifest.Variant): String? {
        if (manifest?.isEstimated(v) != false) return "This Smaug-Decisions variant cannot be downloaded yet (its size is an estimate)."
        return ModelStorage.shortfall(ModelStorage.downloadNeed(v, partBytes(v)), freeBytes(), "Smaug-Decisions")
    }

    fun startDownload(wifiOnly: Boolean): String? {
        val v = pendingDownload ?: return "Nothing to download."
        downloadBlocker(v)?.let { return it }
        val m = manifest!!
        val req = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(workDataOf("file" to v.file, "url" to m.url(v), "sha256" to v.sha256, "bytes" to v.bytes))
            .setConstraints(Constraints.Builder().setRequiredNetworkType(if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED).build())
            .build()
        WorkManager.getInstance(ctx).enqueueUniqueWork(WORK, ExistingWorkPolicy.KEEP, req)
        return null
    }

    /** stops the download; the .part file stays, so the next start resumes */
    fun pauseDownload() { WorkManager.getInstance(ctx).cancelUniqueWork(WORK) }

    fun downloadFlow() = WorkManager.getInstance(ctx).getWorkInfosForUniqueWorkFlow(WORK)

    /** Switches the models; other variants' files are deleted only when [deleteOthers] (the user confirmed). */
    fun apply(c: ModelChoice, deleteOthers: Boolean): String? {
        if (c !in offered) return null
        val p = plan(c)
        prefs.edit().putString(KEY_CHOICE, c.id).apply()
        if (!deleteOthers) return null
        p.delete.forEach { File(dir, it.file).delete(); File(dir, it.file + ".part").delete() }
        val freed = ModelStorage.dropStaleCaches(cacheDir, installedFiles, dir)
        return "Deleted ${p.delete.size} file(s), freed ${ModelStorage.gb(p.freedBytes + freed)}."
    }

    fun isPrepared(v: ModelManifest.Variant) = ModelPrep.isPrepared(prepStore, v.file, v.bytes, backendFor(v), ModelStorage.cacheBytes(cacheDir, v.file))
    fun expectedPrepareSeconds(v: ModelManifest.Variant) = ModelPrep.expectedSeconds(prepStore, v.bytes, backendFor(v))

    /** why Smaug-Decisions cannot take a [kind] decision now because of its file or storage, or null */
    fun fileBlocker(kind: String): String? {
        val v = activeVariant ?: return if (downloadPercent >= 0) "Smaug-Decisions is downloading ($downloadPercent%)" else "Smaug-Decisions not downloaded yet"
        // a file only loads once complete (an adb push or copy that stopped midway has the wrong size)
        if (File(dir, v.file).length() != v.bytes) return "Smaug-Decisions file is incomplete"
        val vision = kind == "image" && v.hasImage
        val existing = ModelStorage.cacheBytes(cacheDir, v.file)
        val toWrite = (if (vision) v.cacheBytes else v.cacheBytesWithoutVision) - existing
        if (!isPrepared(v) || toWrite > 100_000_000) {
            ModelStorage.shortfall(ModelStorage.firstLoadNeed(v, existing, vision), freeBytes(),
                if (isPrepared(v)) "Attaching Smaug-Decisions' vision encoder" else "Preparing Smaug-Decisions")?.let { return it }
        }
        return null
    }

    companion object {
        const val WORK = "model-download"
        const val KEY_CHOICE = "models.choice"
        const val BACKEND = "cpu"
    }
}

/**
 * The download as a WorkManager worker: a foreground dataSync job with a progress notification, so it keeps going when the app
 * is in the background; [ResumableDownloader] continues a partial file after a pause, a lost connection or a process death.
 */
class ModelDownloadWorker(ctx: Context, params: WorkerParameters) : Worker(ctx, params) {
    @Volatile private var dl: ResumableDownloader? = null

    override fun doWork(): Result {
        val item = ResumableDownloader.Item(inputData.getString("file")!!, inputData.getString("url")!!, inputData.getString("sha256")!!,
            inputData.getLong("bytes", 0))
        val d = ResumableDownloader(item, applicationContext.getExternalFilesDir(null)!!); dl = d
        runCatching { setForegroundAsync(foreground(0, item.bytes)).get() }
        var lastPct = -1
        val r = d.run({ got, total ->
            val pct = (got * 100 / maxOf(total, 1)).toInt()
            if (pct != lastPct) { lastPct = pct; setProgressAsync(workDataOf("received" to got, "total" to total, "phase" to "downloading")); runCatching { setForegroundAsync(foreground(got, total)) } }
        }, { f -> setProgressAsync(workDataOf("received" to item.bytes, "total" to item.bytes, "phase" to "verifying", "verify" to f)) })
        return when (r) {
            is ResumableDownloader.Result.Done -> Result.success(workDataOf("file" to r.file.name))
            is ResumableDownloader.Result.Cancelled -> Result.failure(workDataOf("message" to "Paused at ${ModelStorage.gb2(r.received)} of ${ModelStorage.gb2(item.bytes)}."))
            is ResumableDownloader.Result.Failed -> if (r.retryable && runAttemptCount < 5) Result.retry() else Result.failure(workDataOf("message" to r.message))
        }
    }

    override fun onStopped() { dl?.cancel() }

    private fun foreground(got: Long, total: Long): ForegroundInfo {
        val nm = applicationContext.getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) nm.createNotificationChannel(NotificationChannel(CHANNEL, "Smaug-Decisions download", NotificationManager.IMPORTANCE_LOW))
        val n = NotificationCompat.Builder(applicationContext, CHANNEL).setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Downloading Smaug-Decisions").setContentText("${ModelStorage.gb2(got)} of ${ModelStorage.gb2(total)}")
            .setProgress(1000, (got * 1000 / maxOf(total, 1)).toInt(), false).setOngoing(true).setOnlyAlertOnce(true).build()
        return if (Build.VERSION.SDK_INT >= 29) ForegroundInfo(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC) else ForegroundInfo(NOTIFICATION_ID, n)
    }

    companion object {
        const val CHANNEL = "model_download"
        const val NOTIFICATION_ID = 4711
    }
}

/** What the UI shows of the download, from the unique work's WorkInfo. */
data class DownloadUi(val phase: String, val received: Long = 0, val total: Long = 0, val verify: Double = 0.0, val message: String = "") {
    val percent get() = if (total > 0) (received * 100 / total).toInt() else 0
    val active get() = phase == "downloading" || phase == "verifying" || phase == "queued"

    companion object {
        fun of(infos: List<WorkInfo>): DownloadUi {
            val w = infos.lastOrNull() ?: return DownloadUi("idle")
            val p: Data = w.progress
            return when (w.state) {
                WorkInfo.State.ENQUEUED, WorkInfo.State.BLOCKED -> DownloadUi("queued", message = "Waiting for a network connection.")
                WorkInfo.State.RUNNING -> DownloadUi(p.getString("phase") ?: "downloading", p.getLong("received", 0), p.getLong("total", 0), p.getDouble("verify", 0.0))
                WorkInfo.State.SUCCEEDED -> DownloadUi("done", message = w.outputData.getString("file") ?: "")
                WorkInfo.State.FAILED -> DownloadUi("failed", message = w.outputData.getString("message") ?: "The download failed.")
                WorkInfo.State.CANCELLED -> DownloadUi("paused", message = "Download paused. It resumes where it stopped.")
            }
        }
    }
}
