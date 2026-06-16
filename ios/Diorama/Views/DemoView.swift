import SwiftUI
import SceneKit

/// Demo mode (launch arg `-DioramaDemo 1`): downloads the real pipeline output
/// (textured GLB + photo backdrop + optional depth map + meta.json) from a
/// runner-local server, builds the bounded-orbit viewer with the real photo
/// camera, and auto-orbits — so CI can record the actual output.
struct DemoView: View {
    @State private var controller: DioramaSceneController?
    @State private var status = "Loading sample diorama…"
    @State private var failed = false

    private let api = DioramaAPI()

    /// meta.json emitted by the backend CLI.
    private struct Meta: Codable {
        let num_people: Int?
        let img_w: Int
        let img_h: Int
        let avg_cam_tz: Double
        let focal_length: Double
        let depth_available: Bool?
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let controller {
                DemoSceneContainer(view: controller.scnView).ignoresSafeArea()
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
                    Text(status).foregroundColor(.white)
                        .multilineTextAlignment(.center).padding(.horizontal, 32)
                }
            }
        }
        .task { await load() }
    }

    private func load() async {
        do {
            status = "Downloading sample assets…"
            let glbURL = AppConfig.demoGLBURL
            let base = glbURL.deletingLastPathComponent()

            let glb = try await api.download(glbURL, suggestedName: "demo.glb")
            let backdropFile = try await api.download(AppConfig.demoBackdropURL, suggestedName: "demo.jpg")
            guard let image = UIImage(contentsOfFile: backdropFile.path) else {
                throw DioramaAPI.ClientError.badResponse
            }

            // Pull the real camera metadata so the opening frame matches the photo.
            let meta = await fetchMeta(base.appendingPathComponent("meta.json"))

            // Optional depth map → depth-mesh background.
            var depthImage: UIImage?
            if meta?.depth_available == true {
                if let depthFile = try? await api.download(base.appendingPathComponent("depth.png"), suggestedName: "depth.png") {
                    depthImage = UIImage(contentsOfFile: depthFile.path)
                }
            }

            let hint = DioramaResult(
                aligned: false,
                hero_glb_url: "",
                backdrop: .init(image_url: "",
                                img_w: meta?.img_w ?? Int(image.size.width),
                                img_h: meta?.img_h ?? Int(image.size.height)),
                scene_hint: .init(num_people: meta?.num_people ?? 1,
                                  avg_cam_tz: meta?.avg_cam_tz ?? 3.0,
                                  focal_length: meta?.focal_length ?? 1000,
                                  people_bboxes: [])
            )

            let ctrl = try DioramaSceneController(heroGLB: glb, backdropImage: image,
                                                  depthImage: depthImage, hint: hint)
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

    private func fetchMeta(_ url: URL) async -> Meta? {
        guard let (data, _) = try? await URLSession.shared.data(from: url) else { return nil }
        return try? JSONDecoder().decode(Meta.self, from: data)
    }
}

private struct DemoSceneContainer: UIViewRepresentable {
    let view: SCNView
    func makeUIView(context: Context) -> SCNView { view }
    func updateUIView(_ uiView: SCNView, context: Context) {}
}
