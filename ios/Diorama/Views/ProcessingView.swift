import SwiftUI

/// Processing screen (spec §3.2 screen 2): upload the still, poll every 2 s,
/// show the advisory stage label, then download the assets for the viewer.
struct ProcessingView: View {
    let jpeg: Data
    let onReady: (ViewerAssets) -> Void
    let onCancel: () -> Void

    @State private var stageLabel = "Uploading photo…"
    @State private var errorMessage: String?
    private let api = DioramaAPI()

    var body: some View {
        VStack(spacing: 24) {
            if let errorMessage {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 44)).foregroundColor(.orange)
                Text(errorMessage)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.white)
                    .padding(.horizontal, 32)
                Button("Try another photo", action: onCancel)
                    .buttonStyle(.borderedProminent)
            } else {
                ProgressView().scaleEffect(1.6).tint(.white)
                Text(stageLabel)
                    .foregroundColor(.white)
                    .font(.headline)
                Text("This usually takes 30–90 seconds.")
                    .foregroundColor(.white.opacity(0.6))
                    .font(.footnote)
            }
        }
        .task { await run() }
    }

    private func run() async {
        do {
            let jobId = try await api.submit(jpeg: jpeg)
            let result = try await api.awaitResult(jobId: jobId) { stage, _ in
                Task { @MainActor in stageLabel = Self.label(for: stage) }
            }

            await MainActor.run { stageLabel = "Downloading scene…" }
            guard let heroURL = URL(string: result.hero_glb_url),
                  let backdropURL = URL(string: result.backdrop.image_url) else {
                throw DioramaAPI.ClientError.badResponse
            }
            let heroLocal = try await api.download(heroURL, suggestedName: "hero.glb")
            let backdropLocal = try await api.download(backdropURL, suggestedName: "backdrop.jpg")
            guard let img = UIImage(contentsOfFile: backdropLocal.path) else {
                throw DioramaAPI.ClientError.badResponse
            }

            let assets = ViewerAssets(heroGLB: heroLocal, backdrop: img, result: result)
            await MainActor.run { onReady(assets) }
        } catch {
            await MainActor.run { errorMessage = Self.friendly(error) }
        }
    }

    /// Map the advisory stage string (spec §2.3) to user-facing copy.
    private static func label(for stage: String?) -> String {
        switch stage {
        case "uploading": return "Uploading photo…"
        case "segmenting": return "Analyzing the photo…"
        case "reconstructing_bodies": return "Reconstructing people…"
        case "building_backdrop": return "Placing the backdrop…"
        case "aligning": return "Aligning the scene…"
        case "packaging": return "Finishing up…"
        default: return "Building your diorama…"
        }
    }

    /// Non-technical error copy (spec §3.6).
    private static func friendly(_ error: Error) -> String {
        if case let DioramaAPI.ClientError.server(api) = error {
            switch api.code {
            case "NO_PEOPLE_DETECTED":
                return "Couldn't find people in the photo. Make sure full bodies are visible and try again."
            case "RECON_FAILED", "TIMEOUT", "INTERNAL":
                return "Something went wrong building your diorama. Try another photo."
            default:
                return api.message
            }
        }
        return "Something went wrong. Please try again."
    }
}
