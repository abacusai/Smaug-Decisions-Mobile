import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// path to a llama.cpp checkout (tested with b11979); -PllamaCppDir=... or llamaCppDir=... in local.properties. Only the native
// build needs it (CMake stops with "pass -DLLAMA_CPP_DIR" when it is missing); the JVM unit tests do not.
val llamaCppDir: String = (project.findProperty("llamaCppDir") as String?)
    ?: Properties().apply { rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) } }
        .getProperty("llamaCppDir", null) ?: ""

val fastAssets: String = (project.findProperty("fastAssets") as String?)
    ?: Properties().apply { rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) } }
        .getProperty("fastAssets", null) ?: ""

fun localProp(k: String): String? = (project.findProperty(k) as String?)
    ?: Properties().apply { rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) } }.getProperty(k, null)

// Smaug-Decisions backend: "litert" (default; LiteRT-LM with our harness patch, prebuilt by runtime/build_litert.sh into litertDir:
// include/c/*.h, android_arm64/*.so, linux_x86_64/*.so) or "llamacpp" (fallback; needs llamaCppDir)
val bigBackend: String = localProp("bigBackend") ?: "litert"
val litertDir: String = localProp("litertDir") ?: ""

// optional: keep the (large) build outputs off the source disk: buildRoot=<dir> in local.properties
localProp("buildRoot")?.let { layout.buildDirectory.set(file("$it/app")) }

val ortVersion = "1.23.2"   // same version as the Python onnxruntime used for fast/parity_onnx.py

val abis: List<String> = ((project.findProperty("abis") as String?) ?: if (bigBackend == "litert") "arm64-v8a" else "arm64-v8a,x86_64").split(",").map { it.trim() }.filter { it.isNotEmpty() }

android {
    namespace = "ai.abacus.decide"
    compileSdk = 36
    ndkVersion = "29.0.13113456"

    defaultConfig {
        applicationId = "ai.abacus.decide"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "BIG_BACKEND", "\"$bigBackend\"")
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DDECIDE_BIG_BACKEND=$bigBackend",
                    "-DLITERT_DIR=$litertDir",
                    "-DLLAMA_CPP_DIR=$llamaCppDir",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DANDROID_STL=c++_shared",
                )
            }
        }
    }
    externalNativeBuild {
        cmake {
            localProp("buildRoot")?.let { buildStagingDirectory = file("$it/cxx") }
            path("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
    buildTypes {
        // native code is always built Release (above); the debug APK only differs in signing / debuggable
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { buildConfig = true; compose = true }
    // LiteRT-LM runtime libraries (liblitert-lm.so and the prebuilt libs it links) come from litertDir
    if (bigBackend == "litert" && litertDir.isNotEmpty()) sourceSets["main"].jniLibs.srcDir("$litertDir/jniLibs")
    packaging {
        jniLibs {
            useLegacyPackaging = false; pickFirsts += listOf("**/liblitert-lm.so")
            // only the ABIs we build natively (the ORT AAR ships four; the LiteRT-LM runtime is arm64-only)
            excludes += listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64").filter { it !in abis }.map { "lib/$it/**" }
        }
    }
    // one APK per ABI: arm64-v8a (phones) and x86_64 (emulators / Chromebooks); -Pabis=arm64-v8a builds just one
    splits {
        abi {
            isEnable = true
            reset()
            include(*abis.toTypedArray())
            isUniversalApk = false
        }
    }
    // self-test media (assets/selftest) stored uncompressed
    androidResources { noCompress += listOf("wav", "png", "jpg", "json", "onnx", "bin") }
    testOptions { unitTests.all { it.configureHostTests() } }
    // the fast tier's model files (fast/audio.onnx, fast/trunk.onnx, fast/embed_tokens_int8.bin + _scales.bin, fast/head.json; ~600 MB, not in
    // git): -PfastAssets=<dir containing fast/> or fastAssets=... in local.properties. Without them the APK builds but the app
    // reports the fast tier missing.
    // -PfastPack=true: the model files go into the :fastpack install-time asset pack (Play Asset Delivery, for bundles)
    if (localProp("fastPack") == "true") assetPacks += listOf(":fastpack")
    else sourceSets["main"].assets.srcDir(fastAssets.ifEmpty { "src/main/assets-none" })
}

dependencies {
    // fast tier: ONNX Runtime (CPU EP; XNNPACK optional). The desktop jar runs the same Kotlin FastTier in the JVM host tests.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion")
    // UI: Jetpack Compose (Material 3), styled after the ChatLLM Android app; background download: WorkManager
    val composeBom = platform("androidx.compose:compose-bom:2026.05.00")
    implementation(composeBom)
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.12.4")
    implementation("androidx.work:work-runtime:2.11.2")
    implementation("androidx.core:core-ktx:1.17.0")
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")   // android.jar's org.json is a stub on the JVM
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

/**
 * JVM unit tests (./gradlew test) run the pure-Kotlin goldens always. The host end-to-end test (HostE2ETest: the same Decider +
 * JNI + C core + llama.cpp, built for the desktop) runs only when -Pdecide.hostLib=<dir with libdecide_jni.so> is given, with
 * -Pdecide.model / -Pdecide.mmproj (GGUF paths) and optionally -Pdecide.threads, -Pdecide.only=c00,c11, -Pdecide.report=<json>.
 * See ../run_host_e2e.sh.
 */
fun Test.configureHostTests() {
    maxHeapSize = (project.findProperty("decide.heap") as String?) ?: "2g"
    testLogging { events("passed", "skipped", "failed", "standardOut", "standardError"); showStandardStreams = true }
    listOf("decide.hostLib", "decide.model", "decide.mmproj", "decide.threads", "decide.only", "decide.report", "decide.bundle",
        "decide.fastDir", "decide.cascadeBundle", "decide.tau",
        "decide.bigBackend", "decide.litertModel", "decide.litertCache", "decide.litertTmp", "decide.litertVision",
        "decide.litertKeepVision", "decide.parityClips", "decide.liveDownload").forEach { k ->
        project.findProperty(k)?.let { systemProperty(k, it.toString()) }
    }
    project.findProperty("decide.hostLib")?.let { systemProperty("java.library.path", it.toString()) }
    // the phone configuration: llama.cpp keeps the weights in the mmapped file instead of repacked anonymous copies
    project.findProperty("decide.noRepack")?.let { environment("DEC_NO_REPACK", it.toString()) }
    if (!project.hasProperty("decide.bundle")) systemProperty("decide.bundle", project.file("src/main/assets/selftest").absolutePath)
    outputs.upToDateWhen { false }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
