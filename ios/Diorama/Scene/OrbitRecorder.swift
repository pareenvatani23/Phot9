import AVFoundation
import SceneKit
import UIKit

/// Records the auto-orbit sweep to an MP4 (spec §3.5): azimuth 0 → −0.9 → +0.9
/// → 0 over ~5 s (ease-in-out), each frame snapshotted from the live SCNView
/// and written via AVAssetWriter (H.264, 30 fps, 1080×1920 portrait).
final class OrbitRecorder {
    private let controller: DioramaSceneController
    private let duration: TimeInterval = 5.0
    private let fps: Int32 = 30
    private let renderSize = CGSize(width: 1080, height: 1920)

    // Azimuth keyframe arc (radians) — stays inside the ±70° clamp.
    private let sweepLeft: Float = -0.9
    private let sweepRight: Float = 0.9

    init(controller: DioramaSceneController) {
        self.controller = controller
    }

    enum RecorderError: Error { case setupFailed }

    /// Render the sweep and call back with the finished MP4 file URL.
    func record(completion: @escaping (Result<URL, Error>) -> Void) {
        let totalFrames = Int(Double(fps) * duration)
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("diorama-orbit-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outURL)

        guard let writer = try? AVAssetWriter(outputURL: outURL, fileType: .mp4) else {
            completion(.failure(RecorderError.setupFailed)); return
        }

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: renderSize.width,
            AVVideoHeightKey: renderSize.height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: renderSize.width,
            kCVPixelBufferHeightKey as String: renderSize.height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)

        guard writer.canAdd(input) else { completion(.failure(RecorderError.setupFailed)); return }
        writer.add(input)
        guard writer.startWriting() else { completion(.failure(writer.error ?? RecorderError.setupFailed)); return }
        writer.startSession(atSourceTime: .zero)

        let queue = DispatchQueue(label: "diorama.recorder")
        var frame = 0

        input.requestMediaDataWhenReady(on: queue) { [weak self] in
            guard let self else { return }
            while input.isReadyForMoreMediaData {
                if frame >= totalFrames {
                    input.markAsFinished()
                    writer.finishWriting {
                        DispatchQueue.main.async {
                            self.controller.resetToFront()
                            if writer.status == .completed {
                                completion(.success(outURL))
                            } else {
                                completion(.failure(writer.error ?? RecorderError.setupFailed))
                            }
                        }
                    }
                    return
                }

                let t = Double(frame) / Double(totalFrames - 1)
                let azimuth = self.azimuth(at: Float(t))

                // Snapshot must happen on the main thread with the camera posed.
                let image: UIImage = DispatchQueue.main.sync {
                    self.controller.setAzimuth(azimuth)
                    return self.controller.scnView.snapshot()
                }

                if let buffer = self.pixelBuffer(from: image, pool: adaptor.pixelBufferPool) {
                    let time = CMTime(value: CMTimeValue(frame), timescale: self.fps)
                    adaptor.append(buffer, withPresentationTime: time)
                }
                frame += 1
            }
        }
    }

    // MARK: Easing

    /// Piecewise smoothstep: 0 →(¼)→ left →(½)→ right →(¼)→ 0.
    private func azimuth(at t: Float) -> Float {
        switch t {
        case ..<0.25:
            return lerpSmooth(0, sweepLeft, t / 0.25)
        case ..<0.75:
            return lerpSmooth(sweepLeft, sweepRight, (t - 0.25) / 0.5)
        default:
            return lerpSmooth(sweepRight, 0, (t - 0.75) / 0.25)
        }
    }

    private func lerpSmooth(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let c = min(1, max(0, x))
        let s = c * c * (3 - 2 * c)   // smoothstep ease-in-out
        return a + (b - a) * s
    }

    // MARK: Pixel buffer

    /// Render `image` aspect-fill into a pool pixel buffer at `renderSize`.
    private func pixelBuffer(from image: UIImage, pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        guard let buffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: Int(renderSize.width),
            height: Int(renderSize.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        ), let cg = image.cgImage else { return nil }

        ctx.setFillColor(UIColor.black.cgColor)
        ctx.fill(CGRect(origin: .zero, size: renderSize))

        // Aspect-fill the snapshot into the portrait frame.
        let iw = CGFloat(cg.width), ih = CGFloat(cg.height)
        let scale = max(renderSize.width / iw, renderSize.height / ih)
        let dw = iw * scale, dh = ih * scale
        let rect = CGRect(x: (renderSize.width - dw) / 2, y: (renderSize.height - dh) / 2, width: dw, height: dh)
        ctx.draw(cg, in: rect)
        return buffer
    }
}
