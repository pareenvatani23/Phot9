import SwiftUI
import SceneKit

/// Viewer screen (spec §3.2 screen 3 / §3.4 / §3.5): bounded-orbit SceneKit
/// view with a "Record orbit" button that exports an MP4 to the share sheet.
struct ViewerView: View {
    let assets: ViewerAssets
    let onClose: () -> Void

    @State private var controller: DioramaSceneController?
    @State private var loadError: String?
    @State private var recording = false
    @State private var shareURL: URL?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let controller {
                SceneContainer(view: controller.scnView)
                    .ignoresSafeArea()

                VStack {
                    HStack {
                        Button(action: onClose) {
                            Image(systemName: "xmark")
                                .font(.title3.weight(.semibold))
                                .foregroundColor(.white)
                                .padding(10)
                                .background(.black.opacity(0.4), in: Circle())
                        }
                        Spacer()
                    }
                    .padding()
                    Spacer()
                    recordButton(controller)
                        .padding(.bottom, 40)
                }
            } else if let loadError {
                VStack(spacing: 16) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 44)).foregroundColor(.orange)
                    Text(loadError).foregroundColor(.white).multilineTextAlignment(.center)
                    Button("Back", action: onClose).buttonStyle(.borderedProminent)
                }.padding()
            } else {
                ProgressView().tint(.white)
            }
        }
        .onAppear(perform: build)
        .sheet(item: $shareURL) { url in
            ActivityView(items: [url])
        }
    }

    private func recordButton(_ controller: DioramaSceneController) -> some View {
        Button {
            recording = true
            OrbitRecorder(controller: controller).record { result in
                DispatchQueue.main.async {
                    recording = false
                    if case .success(let url) = result { shareURL = url }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: recording ? "hourglass" : "record.circle")
                Text(recording ? "Recording…" : "Record orbit")
            }
            .font(.headline)
            .foregroundColor(.white)
            .padding(.horizontal, 22).padding(.vertical, 14)
            .background(.red.opacity(0.85), in: Capsule())
        }
        .disabled(recording)
    }

    private func build() {
        guard controller == nil, loadError == nil else { return }
        do {
            controller = try DioramaSceneController(
                heroGLB: assets.heroGLB,
                backdropImage: assets.backdrop,
                hint: assets.result
            )
        } catch {
            loadError = "Couldn't load the 3D scene."
        }
    }
}

/// Wraps the controller's SCNView for SwiftUI.
private struct SceneContainer: UIViewRepresentable {
    let view: SCNView
    func makeUIView(context: Context) -> SCNView { view }
    func updateUIView(_ uiView: SCNView, context: Context) {}
}

/// System share sheet for the exported MP4.
struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

/// Allow `URL` to drive `.sheet(item:)`.
extension URL: Identifiable {
    public var id: String { absoluteString }
}
