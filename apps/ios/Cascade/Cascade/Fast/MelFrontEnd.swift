// Log-mel front end matching transformers' Gemma4AudioFeatureExtractor defaults (EmbeddingGemma 2 / Gemma 4 audio towers):
// 16 kHz mono, 160 zeros prepended, frames of 320 samples every 160, periodic Hann, rFFT 512 (magnitude, not power),
// 128 HTK mel filters 0-8 kHz without normalisation, log(mel + 1e-3). Frame i is valid when sample i*160 + 320 (the last sample
// of the 321-sample unfold window) is real audio; invalid frames are zeroed. Output: [frames][128] row-major Float32.
import Accelerate
import Foundation

public final class MelFrontEnd {
    public static let sampleRate = 16000, frameLength = 320, hop = 160, fftLength = 512, nMels = 128, melFloor: Float = 1e-3
    private let window: [Float]
    private let melFilters: [Float]   // [257 * 128], row = frequency bin
    private let fft: vDSP.FFT<DSPSplitComplex>
    private let log2n = vDSP_Length(9)

    public init() {
        var w = [Float](repeating: 0, count: MelFrontEnd.frameLength)
        for i in 0..<w.count { w[i] = 0.5 - 0.5 * cos(2 * Float.pi * Float(i) / Float(w.count)) }   // periodic Hann (np.hanning(N+1)[:-1])
        window = w
        fft = vDSP.FFT(log2n: 9, radix: .radix2, ofType: DSPSplitComplex.self)!
        melFilters = MelFrontEnd.htkMelFilterBank(nFreq: 257, nMels: 128, sampleRate: 16000, fMin: 0, fMax: 8000)
    }

    /// transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk"): triangular filters on the HTK mel scale.
    static func htkMelFilterBank(nFreq: Int, nMels: Int, sampleRate: Int, fMin: Float, fMax: Float) -> [Float] {
        func hzToMel(_ f: Float) -> Float { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Float) -> Float { 700 * (pow(10, m / 2595) - 1) }
        let melMin = hzToMel(fMin), melMax = hzToMel(fMax)
        let melPts = (0..<(nMels + 2)).map { melMin + (melMax - melMin) * Float($0) / Float(nMels + 1) }
        let hzPts = melPts.map(melToHz)
        let fftFreqs = (0..<nFreq).map { Float($0) * Float(sampleRate) / Float(2 * (nFreq - 1)) }
        var fb = [Float](repeating: 0, count: nFreq * nMels)
        for m in 0..<nMels {
            let lo = hzPts[m], c = hzPts[m + 1], hi = hzPts[m + 2]
            for k in 0..<nFreq {
                let f = fftFreqs[k]
                let up = (f - lo) / (c - lo), down = (hi - f) / (hi - c)
                fb[k * nMels + m] = max(0, min(up, down))
            }
        }
        return fb
    }

    /// Number of mel frames the extractor produces for n samples (no 128-multiple padding applied here).
    public static func frameCount(samples n: Int) -> Int { max(0, (n + frameLength / 2 - (frameLength + 1)) / hop + 1) }

    /// samples: Float32 PCM at 16 kHz. Returns (features [frames*128], mask [frames]) for exactly `frames` frames (zero-padded / cut).
    public func features(samples: [Float], frames: Int) -> (features: [Float], mask: [Float]) {
        let padLeft = MelFrontEnd.frameLength / 2
        var x = [Float](repeating: 0, count: padLeft + samples.count); x.replaceSubrange(padLeft..<x.count, with: samples)
        let nValidFrames = min(frames, MelFrontEnd.frameCount(samples: samples.count))
        var feats = [Float](repeating: 0, count: frames * MelFrontEnd.nMels)
        var mask = [Float](repeating: 0, count: frames)
        var re = [Float](repeating: 0, count: MelFrontEnd.fftLength / 2), im = [Float](repeating: 0, count: MelFrontEnd.fftLength / 2)
        var frame = [Float](repeating: 0, count: MelFrontEnd.fftLength)
        var mag = [Float](repeating: 0, count: MelFrontEnd.fftLength / 2 + 1)
        var mel = [Float](repeating: 0, count: MelFrontEnd.nMels)
        for i in 0..<nValidFrames {
            let start = i * MelFrontEnd.hop
            for j in 0..<MelFrontEnd.fftLength { frame[j] = 0 }
            for j in 0..<MelFrontEnd.frameLength where start + j < x.count { frame[j] = x[start + j] * window[j] }
            // real FFT via split-complex packing (vDSP convention: even samples -> real, odd -> imag; output packed)
            frame.withUnsafeBufferPointer { fp in
                re.withUnsafeMutableBufferPointer { rp in im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    fp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: MelFrontEnd.fftLength / 2) { vDSP_ctoz($0, 2, &split, 1, vDSP_Length(MelFrontEnd.fftLength / 2)) }
                    fft.forward(input: split, output: &split)
                } }
            }
            // unpack: bin 0 = re[0]/2 (DC), bin N/2 = im[0]/2 (Nyquist); other bins (re[k]/2, im[k]/2) -> magnitude
            mag[0] = abs(re[0]) / 2; mag[MelFrontEnd.fftLength / 2] = abs(im[0]) / 2
            for k in 1..<(MelFrontEnd.fftLength / 2) { mag[k] = sqrt(re[k] * re[k] + im[k] * im[k]) / 2 }
            vDSP_mmul(mag, 1, melFilters, 1, &mel, 1, 1, vDSP_Length(MelFrontEnd.nMels), vDSP_Length(MelFrontEnd.fftLength / 2 + 1))
            for m in 0..<MelFrontEnd.nMels { feats[i * MelFrontEnd.nMels + m] = log(mel[m] + MelFrontEnd.melFloor) }
            mask[i] = 1
        }
        return (feats, mask)
    }
}
