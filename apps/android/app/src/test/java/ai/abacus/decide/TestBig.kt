package ai.abacus.decide

import java.io.File

/** Picks the Smaug-Decisions backend for the host tests: -Pdecide.bigBackend=litert (with -Pdecide.litertModel, -Pdecide.litertCache) or
 *  llamacpp (default; -Pdecide.model / -Pdecide.mmproj). -Pdecide.hostLib is the dir with the host JNI library. */
object TestBig {
    val backend: String get() = System.getProperty("decide.bigBackend") ?: "llamacpp"
    val configured: Boolean get() = System.getProperty("decide.hostLib") != null && when (backend) {
        "litert" -> System.getProperty("decide.litertModel") != null
        else -> System.getProperty("decide.model") != null && System.getProperty("decide.mmproj") != null
    }

    fun create(b: Double, threads: Int): BigModel = when (backend) {
        "litert" -> LiteRtDecider(System.getProperty("decide.litertModel"),
            File(System.getProperty("decide.litertCache") ?: (System.getProperty("decide.litertModel") + ".cache")),
            File(System.getProperty("decide.litertTmp") ?: System.getProperty("java.io.tmpdir")), threads = threads, b = b,
            vision = LiteRtDecider.Vision.valueOf(System.getProperty("decide.litertVision") ?: "ON_DEMAND"),
            keepVision = System.getProperty("decide.litertKeepVision") != "false")
        else -> Decider(System.getProperty("decide.model"), System.getProperty("decide.mmproj"), b, threads = threads)
    }
}
