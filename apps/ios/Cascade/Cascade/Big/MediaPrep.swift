import AVFoundation
import UIKit

/// Media in the form the model was trained on, mirroring smaug_decide.load_audio and the Image.convert('RGB') path.
enum MediaPrep {
    static let tmp = FileManager.default.temporaryDirectory

    /// Any audio file -> mono (channel mean) 16 kHz (linear interpolation, like the reference) float32 WAV, last 30 s.
    /// Returns the WAV and its number of samples.
    static func wav16kMono(from src: URL, name: String) throws -> (URL, Int) {
        let samples = try samples16kMono(from: src)
        return (try wav(samples, name: name), samples.count)
    }

    /// Any audio file -> mono 16 kHz float32 samples, last 30 s (smaug_decide.load_audio).
    static func samples16kMono(from src: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: src)
        let format = file.processingFormat   // float32, deinterleaved; int16 input is scaled by 1/32768 like soundfile
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw DeciderError(message: "The recording is empty.")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { throw DeciderError(message: "Unsupported audio format.") }
        let count = Int(buffer.frameLength), nch = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: count)
        for c in 0..<nch {
            for i in 0..<count { mono[i] += channels[c][i] }
        }
        if nch > 1 { mono = mono.map { $0 / Float(nch) } }
        var samples = format.sampleRate == Double(Prompt.sampleRate) ? mono : resampleLinear(mono, from: format.sampleRate)
        samples = Array(samples.suffix(Prompt.maxAudioSeconds * Prompt.sampleRate))
        guard !samples.isEmpty else { throw DeciderError(message: "The recording is empty.") }
        return samples
    }

    /// 16 kHz mono float32 samples -> a WAV in tmp (what the runtime reads), last 30 s.
    static func wav(_ samples: [Float], name: String) throws -> URL {
        let clip = Array(samples.suffix(Prompt.maxAudioSeconds * Prompt.sampleRate))
        guard !clip.isEmpty else { throw DeciderError(message: "The recording is empty.") }
        let url = tmp.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: url)
        try writeWav(clip, to: url)
        return url
    }

    /// np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w) with n = round(len(w) * 16000 / sr).
    static func resampleLinear(_ w: [Float], from sr: Double) -> [Float] {
        let n = Int((Double(w.count) * Double(Prompt.sampleRate) / sr).rounded(.toNearestOrEven))
        guard n > 1, w.count > 1 else { return n == 1 ? Array(w.prefix(1)) : [] }
        let step = Double(w.count - 1) / Double(n - 1)
        return (0..<n).map { i in
            let x = i == n - 1 ? Double(w.count - 1) : Double(i) * step
            let j = min(Int(x), w.count - 2)
            let a = Double(w[j]), b = Double(w[j + 1])
            return Float(a + (b - a) * (x - Double(j)))
        }
    }

    private static func writeWav(_ samples: [Float], to url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: Double(Prompt.sampleRate), AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(Prompt.sampleRate), channels: 1,
                                         interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let out = buffer.floatChannelData?[0] else { throw DeciderError(message: "Could not allocate audio.") }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { out.update(from: $0.baseAddress!, count: samples.count) }
        // The file is finalized (header sizes written) when `file` is released at the end of this scope.
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    }

    /// A picked photo -> upright, opaque (RGB) PNG, longest side at most `maxSide` px (the runtime resizes to its 560-token
    /// grid anyway; this only bounds decode memory).
    static func png(from image: UIImage, name: String, maxSide: CGFloat = 1536) throws -> URL {
        let s = image.size
        let k = min(1, maxSide / max(s.width, s.height, 1))
        let size = CGSize(width: (s.width * k).rounded(), height: (s.height * k).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let data = UIGraphicsImageRenderer(size: size, format: format).pngData { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        let url = tmp.appendingPathComponent(name)
        try data.write(to: url, options: .atomic)
        return url
    }
}
