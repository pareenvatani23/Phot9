import SwiftUI
import AVFoundation

/// Capture screen (spec §3.2 screen 1): live camera + a Vision overlay around
/// detected people and a framing hint, plus a shutter button.
struct CaptureView: View {
    let onCapture: (Data) -> Void

    @StateObject private var camera = CameraController()
    @State private var capturing = false

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            GeometryReader { geo in
                ForEach(Array(camera.personBoxes.enumerated()), id: \.offset) { _, box in
                    let rect = viewRect(for: box, in: geo.size)
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.green.opacity(0.9), lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }
            }
            .ignoresSafeArea()

            VStack {
                Text("Get everyone's full body in frame.")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.black.opacity(0.45), in: Capsule())
                    .padding(.top, 16)
                Spacer()
                shutter
                    .padding(.bottom, 40)
            }

            if !camera.authorized {
                Text("Camera access is needed to capture a group photo.")
                    .multilineTextAlignment(.center)
                    .foregroundColor(.white)
                    .padding()
            }
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }

    private var shutter: some View {
        Button {
            guard !capturing else { return }
            capturing = true
            Task {
                do {
                    let data = try await camera.capturePhoto()
                    camera.stop()
                    await MainActor.run { onCapture(data) }
                } catch {
                    capturing = false
                }
            }
        } label: {
            ZStack {
                Circle().stroke(Color.white, lineWidth: 4).frame(width: 76, height: 76)
                Circle().fill(Color.white).frame(width: 62, height: 62)
            }
            .opacity(capturing ? 0.5 : 1)
        }
        .disabled(capturing || !camera.authorized)
    }

    /// Approximate mapping of a Vision bounding box (normalized, origin
    /// bottom-left) to portrait view coordinates for the framing overlay.
    private func viewRect(for box: CGRect, in size: CGSize) -> CGRect {
        let w = size.width, h = size.height
        return CGRect(
            x: box.minX * w,
            y: (1 - box.maxY) * h,
            width: box.width * w,
            height: box.height * h
        )
    }
}

/// Hosts an AVCaptureVideoPreviewLayer.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
