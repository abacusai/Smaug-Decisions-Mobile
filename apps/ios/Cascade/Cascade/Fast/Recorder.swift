import AVFoundation
import Foundation

/// Microphone capture to 16 kHz mono Float32, capped at `maxSeconds`.
final class Recorder: ObservableObject {
    @Published var isRecording = false
    @Published var level: Float = 0
    @Published var seconds: Double = 0
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private(set) var samples: [Float] = []
    let maxSeconds: Double
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!

    init(maxSeconds: Double = 10) { self.maxSeconds = maxSeconds }

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [])
        try session.setActive(true)
        samples.removeAll(keepingCapacity: true); seconds = 0
        let input = engine.inputNode; let fmt = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: fmt, to: target)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { [weak self] buf, _ in self?.consume(buf) }
        engine.prepare(); try engine.start(); isRecording = true
    }

    private func consume(_ buf: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = 16000 / buf.format.sampleRate
        let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(buf.frameLength) * ratio) + 16)!
        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buf
        }
        guard err == nil, let ch = out.floatChannelData?[0] else { return }
        let n = Int(out.frameLength); let chunk = Array(UnsafeBufferPointer(start: ch, count: n))
        var rms: Float = 0; for v in chunk { rms += v * v }; rms = sqrt(rms / Float(max(1, n)))
        DispatchQueue.main.async {
            guard self.isRecording else { return }
            let room = Int(self.maxSeconds * 16000) - self.samples.count
            if room > 0 { self.samples.append(contentsOf: chunk.prefix(room)) }
            self.seconds = Double(self.samples.count) / 16000; self.level = rms
            if room <= chunk.count { self.stop() }
        }
    }

    func stop() {
        guard isRecording else { return }
        engine.inputNode.removeTap(onBus: 0); engine.stop(); isRecording = false; level = 0
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
