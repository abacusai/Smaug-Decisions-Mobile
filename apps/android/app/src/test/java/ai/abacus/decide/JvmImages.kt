package ai.abacus.decide

/**
 * PNG / JPEG -> RGB bytes with the desktop JDK's javax.imageio. Unit tests compile against android.jar, which hides java.desktop,
 * so ImageIO is reached by reflection; at run time the host JVM has it. PNG decoding is lossless (identical to PIL's).
 */
object JvmImages {
    fun decode(bytes: ByteArray): HfImage.Rgb {
        val img = Class.forName("javax.imageio.ImageIO").getMethod("read", java.io.InputStream::class.java).invoke(null, bytes.inputStream())
        requireNotNull(img) { "ImageIO cannot decode" }
        val c = img.javaClass
        val w = c.getMethod("getWidth").invoke(img) as Int; val h = c.getMethod("getHeight").invoke(img) as Int
        val px = c.getMethod("getRGB", Int::class.java, Int::class.java, Int::class.java, Int::class.java, IntArray::class.java, Int::class.java, Int::class.java)
            .invoke(img, 0, 0, w, h, null, 0, w) as IntArray
        val rgb = ByteArray(w * h * 3)
        for (i in px.indices) { val p = px[i]; rgb[3 * i] = (p shr 16).toByte(); rgb[3 * i + 1] = (p shr 8).toByte(); rgb[3 * i + 2] = p.toByte() }
        return HfImage.Rgb(rgb, w, h)
    }
}
