package ai.abacus.decide

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Image preprocessing in front of llama.cpp's gemma4v encoder. llama.cpp never upscales, so we first resize to the grid
 * transformers' Gemma4 processor uses at Prompt.IMAGE_TOKENS soft tokens (560 for Smaug-Decisions), with a bit-exact port of Pillow's BICUBIC resample (what
 * llama_decide.hf_resize uses; HF itself uses torchvision bicubic + antialias, which is near-identical). Pure Kotlin.
 */
object HfImage {
    class Rgb(val rgb: ByteArray, val width: Int, val height: Int) {
        init { require(rgb.size == width * height * 3) { "rgb size ${rgb.size} != ${width}x${height}x3" } }
    }

    /** transformers' Gemma4 get_aspect_ratio_preserving_size (patch 16, pooling 3, Prompt.IMAGE_TOKENS soft tokens) -> (height, width). */
    fun size(height: Int, width: Int, patch: Int = 16, pool: Int = 3, maxSoftTokens: Int = Prompt.IMAGE_TOKENS): Pair<Int, Int> {
        val maxPatches = maxSoftTokens * pool * pool; val side = pool * patch
        val f = sqrt((maxPatches.toLong() * patch * patch).toDouble() / (height.toLong() * width).toDouble())
        var th = floor(f * height / side).toInt() * side; var tw = floor(f * width / side).toInt() * side
        val maxSide = (maxPatches / (pool * pool)) * side
        if (th == 0) { th = side; tw = minOf(floor(width.toDouble() / height).toInt() * side, maxSide) }
        else if (tw == 0) { tw = side; th = minOf(floor(height.toDouble() / width).toInt() * side, maxSide) }
        return th to tw
    }

    /** soft tokens the encoder produces for an image of this (already gridded) size: (h / 48) * (w / 48). */
    fun softTokens(height: Int, width: Int, patch: Int = 16, pool: Int = 3): Int = (height / (patch * pool)) * (width / (patch * pool))

    /** Resize to the HF grid (no-op when already on it), like llama_decide.hf_resize. */
    fun toGrid(img: Rgb): Rgb {
        val (th, tw) = size(img.height, img.width)
        return if (th == img.height && tw == img.width) img else resizeBicubic(img, tw, th)
    }

    // ---- Pillow libImaging/Resample.c (BICUBIC, a = -0.5, 8 bits per channel, 22-bit fixed point), two passes ----
    private const val PRECISION_BITS = 32 - 8 - 2

    private fun bicubic(x0: Double): Double {
        val a = -0.5; val x = if (x0 < 0.0) -x0 else x0
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a
        return 0.0
    }

    private class Coeffs(val ksize: Int, val bounds: IntArray, val kk: IntArray)

    private fun precompute(inSize: Int, outSize: Int): Coeffs {
        val in0 = 0.0f; val in1 = inSize.toFloat()
        val scale = (in1 - in0).toDouble() / outSize
        val filterscale = if (scale < 1.0) 1.0 else scale
        val support = 2.0 * filterscale
        val ksize = ceil(support).toInt() * 2 + 1
        val pre = DoubleArray(outSize * ksize); val bounds = IntArray(outSize * 2)
        for (xx in 0 until outSize) {
            val center = in0 + (xx + 0.5) * scale
            var ww = 0.0
            val ss = 1.0 / filterscale
            var xmin = (center - support + 0.5).toInt(); if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt(); if (xmax > inSize) xmax = inSize
            xmax -= xmin
            for (x in 0 until xmax) { val w = bicubic((x + xmin - center + 0.5) * ss); pre[xx * ksize + x] = w; ww += w }
            for (x in 0 until xmax) if (ww != 0.0) pre[xx * ksize + x] /= ww
            bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax
        }
        val kk = IntArray(pre.size) { val v = pre[it] * (1 shl PRECISION_BITS); if (pre[it] < 0) (-0.5 + v).toInt() else (0.5 + v).toInt() }
        return Coeffs(ksize, bounds, kk)
    }

    private fun clip8(v: Int): Int { val s = v shr PRECISION_BITS; return if (s < 0) 0 else if (s > 255) 255 else s }

    /** Bit-exact Image.resize((outW, outH), Image.BICUBIC) of an RGB image. */
    fun resizeBicubic(img: Rgb, outW: Int, outH: Int): Rgb {
        val inW = img.width; val inH = img.height
        val needH = outW != inW; val needV = outH != inH
        val ch = precompute(inW, outW); val cv = precompute(inH, outH)
        val yFirst = cv.bounds[0]; val yLast = cv.bounds[outH * 2 - 2] + cv.bounds[outH * 2 - 1]
        var src = img.rgb; var srcW = inW; var srcH = inH
        val vb = cv.bounds.copyOf()
        if (needH) {
            for (i in 0 until outH) vb[i * 2] -= yFirst
            val rows = yLast - yFirst
            val tmp = ByteArray(outW * rows * 3)
            for (yy in 0 until rows) {
                val rowOff = (yy + yFirst) * inW * 3
                for (xx in 0 until outW) {
                    val xmin = ch.bounds[xx * 2]; val xmax = ch.bounds[xx * 2 + 1]; val k = xx * ch.ksize
                    var s0 = 1 shl (PRECISION_BITS - 1); var s1 = s0; var s2 = s0
                    for (x in 0 until xmax) {
                        val p = rowOff + (x + xmin) * 3; val c = ch.kk[k + x]
                        s0 += (src[p].toInt() and 0xff) * c; s1 += (src[p + 1].toInt() and 0xff) * c; s2 += (src[p + 2].toInt() and 0xff) * c
                    }
                    val o = (yy * outW + xx) * 3
                    tmp[o] = clip8(s0).toByte(); tmp[o + 1] = clip8(s1).toByte(); tmp[o + 2] = clip8(s2).toByte()
                }
            }
            src = tmp; srcW = outW; srcH = rows
        }
        if (!needV) return Rgb(if (src === img.rgb) src.copyOf() else src, srcW, srcH)
        val bounds = if (needH) vb else cv.bounds
        val out = ByteArray(srcW * outH * 3)
        for (yy in 0 until outH) {
            val ymin = bounds[yy * 2]; val ymax = bounds[yy * 2 + 1]; val k = yy * cv.ksize
            for (xx in 0 until srcW) {
                var s0 = 1 shl (PRECISION_BITS - 1); var s1 = s0; var s2 = s0
                for (y in 0 until ymax) {
                    val p = ((y + ymin) * srcW + xx) * 3; val c = cv.kk[k + y]
                    s0 += (src[p].toInt() and 0xff) * c; s1 += (src[p + 1].toInt() and 0xff) * c; s2 += (src[p + 2].toInt() and 0xff) * c
                }
                val o = (yy * srcW + xx) * 3
                out[o] = clip8(s0).toByte(); out[o + 1] = clip8(s1).toByte(); out[o + 2] = clip8(s2).toByte()
            }
        }
        return Rgb(out, srcW, outH)
    }
}
