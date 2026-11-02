// Play Asset Delivery install-time pack with the fast tier's model files (fast/audio.onnx, fast/trunk.onnx,
// fast/embed_tokens_int8.bin + fast/embed_tokens_int8_scales.bin (the token table as int8 with one fp32 scale per row,
// ../ios/tools/quantize_embed.py), fast/head.json; ~600 MB), so the base module stays within Play's size limit. Used when the app is
// built with -PfastPack=true (bundleRelease / bundleDebug); the files must be in fastpack/src/main/assets/fast/ (a symlink to
// your exported ONNX dir works; it is gitignored). Install-time packs are installed with the app and read through AssetManager
// like base assets.
plugins { id("com.android.asset-pack") }

assetPack {
    packName.set("fastpack")
    dynamicDelivery { deliveryType.set("install-time") }
}

// keep the (large) outputs off the source disk when buildRoot is set in local.properties (as for :app)
java.util.Properties().apply { rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) } }
    .getProperty("buildRoot")?.let { layout.buildDirectory.set(file("$it/fastpack")) }
