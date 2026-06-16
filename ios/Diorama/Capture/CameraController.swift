import AVFoundation
import Vision
import UIKit

/// Owns the AVCaptureSession (spec §3.2 screen 1). `.photo` preset, one high-res
/// still via AVCapturePhotoOutput. A parallel video-data output feeds Vision so
/// we can draw a live "are the bodies in frame" overlay.
final class CameraController: NSObject, ObservableObject {
    let session = AVCaptureSession()

    /// Normalized (0...1, origin bottom-left per Vision) person rectangles.
    @Published var personBoxes: [CGRect] = []
    @Published var authorized = false

    private let photoOutput = AVCapturePhotoOutput()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "diorama.camera.session")
    private let visionQueue = DispatchQueue(label: "diorama.camera.vision")
    private var captureContinuation: CheckedContinuation<Data, Error>?

    private lazy var humanRequest: VNDetectHumanRectanglesRequest = {
        let r = VNDetectHumanRectanglesRequest()
        if #available(iOS 15.0, *) { r.upperBodyOnly = false }
        return r
    }()

    // MARK: - Lifecycle

    func start() {
        requestAccess { [weak self] granted in
            guard let self else { return }
            DispatchQueue.main.async { self.authorized = granted }
            guard granted else { return }
            self.sessionQueue.async {
                self.configureIfNeeded()
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func requestAccess(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: completion(true)
        case .notDetermined: AVCaptureDevice.requestAccess(for: .video, completionHandler: completion)
        default: completion(false)
        }
    }

    private var configured = false
    private func configureIfNeeded() {
        guard !configured else { return }
        configured = true

        session.beginConfiguration()
        session.sessionPreset = .photo

        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
        }

        if session.canAddOutput(photoOutput) {
            photoOutput.maxPhotoQualityPrioritization = .quality
            session.addOutput(photoOutput)
        }

        if session.canAddOutput(videoOutput) {
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: visionQueue)
            session.addOutput(videoOutput)
        }

        session.commitConfiguration()
    }

    // MARK: - Capture

    /// Capture one high-res still and return its JPEG data.
    func capturePhoto() async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            sessionQueue.async {
                self.captureContinuation = continuation
                let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
                settings.photoQualityPrioritization = .quality
                self.photoOutput.capturePhoto(with: settings, delegate: self)
            }
        }
    }
}

extension CameraController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let continuation = captureContinuation
        captureContinuation = nil
        if let error { continuation?.resume(throwing: error); return }
        guard let data = photo.fileDataRepresentation() else {
            continuation?.resume(throwing: DioramaAPI.ClientError.badResponse); return
        }
        continuation?.resume(returning: data)
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        do {
            try handler.perform([humanRequest])
            let boxes = (humanRequest.results ?? []).map { $0.boundingBox }
            DispatchQueue.main.async { self.personBoxes = boxes }
        } catch {
            // Detection is best-effort; ignore transient failures.
        }
    }
}
