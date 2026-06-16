import SwiftUI
import SceneKit

/// Demo mode (launch arg `-DioramaDemo 1`): downloads a real sample GLB + real
/// photo backdrop, builds the bounded-orbit viewer, and auto-orbits — so the
/// app's rendered output is visible without the camera or backend. This is what
/// CI records to a video.
struct DemoView: View {
    @State private var controller: DioramaSceneController?
    @State private var status = "Loading sample diorama…"
    @State private var failed = false

    private let api = DioramaAPI()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let controller {
                DemoSceneContainer(view: controller.scnView)
                    .ignoresSafeArea()
                VStack {
                    Text("Diorama — auto-orbit demo")
                        .font(.footnote.weight(.semibold))
                        .foregroundColor(.white.opacity(0.9))
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(.black.opacity(0.4), in: Capsule())
                        .padding(.top, 14)
                    Spacer()
                }
            } else {
                VStack(spacing: 14) {
                    if !failed { ProgressView().tint(.white) }
                    Text(status)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
            }
        }
        .task { await load() }
    }

    private func load() async {
        do {
            status = "Downloading sample assets…"
            let glb = try await api.download(AppConfig.demoGLBURL, suggestedName: "demo.glb")
            let backdropFile = try await api.download(AppConfig.demoBackdropURL, suggestedName: "demo.jpg")
            guard let image = UIImage(contentsOfFile: backdropFile.path) else {
                throw DioramaAPI.ClientError.badResponse
            }

            // Synthesize a scene hint for the sample (no backend metadata here).
            let hint = DioramaResult(
                aligned: false,
                hero_glb_url: "",
                backdrop: .init(image_url: "",
                                img_w: Int(image.size.width),
                                img_h: Int(image.size.height)),
                scene_hint: .init(num_people: 1,
                                  avg_cam_tz: 3.0,
                                  focal_length: 1000,
                                  people_bboxes: [])
            )

            let ctrl = try DioramaSceneController(heroGLB: glb, backdropImage: image, hint: hint)
            await MainActor.run {
                controller = ctrl
                ctrl.startAutoOrbit(period: 6)
            }
        } catch {
            await MainActor.run {
                failed = true
                status = "Couldn't load the sample diorama.\n\(error.localizedDescription)"
            }
        }
    }
}

private struct DemoSceneContainer: UIViewRepresentable {
    let view: SCNView
    func makeUIView(context: Context) -> SCNView { view }
    func updateUIView(_ uiView: SCNView, context: Context) {}
}
