import Foundation

/// App-wide configuration. The backend base URL is the only thing the app
/// needs to know about the server. The FAL_KEY lives ONLY on the backend and
/// is never present in this binary (spec §0.4, §7).
enum AppConfig {
    /// Base URL of the Diorama backend (the `PUBLIC_BASE_URL` it was deployed with).
    /// Override at launch with `-DioramaBaseURL https://...` or set here.
    static let backendBaseURL: URL = {
        if let raw = UserDefaults.standard.string(forKey: "DioramaBaseURL"),
           let url = URL(string: raw) {
            return url
        }
        // Default for the simulator talking to a backend on the same machine.
        return URL(string: "http://localhost:8080")!
    }()

    /// Poll interval for job status (spec §3.2 screen 2 / §2.5).
    static let pollInterval: TimeInterval = 2.0

    // MARK: - Demo mode
    // Launch with `-DioramaDemo 1` to skip capture/backend and render the
    // bounded-orbit viewer directly from a real sample GLB + real photo
    // backdrop, auto-orbiting. Used by CI to record the viewer's output.
    // Override the assets with `-DioramaDemoGLB <url>` / `-DioramaDemoBackdrop <url>`.

    static var demoMode: Bool { UserDefaults.standard.bool(forKey: "DioramaDemo") }

    static var demoGLBURL: URL {
        let raw = UserDefaults.standard.string(forKey: "DioramaDemoGLB")
            ?? "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb"
        return URL(string: raw)!
    }

    static var demoBackdropURL: URL {
        let raw = UserDefaults.standard.string(forKey: "DioramaDemoBackdrop")
            ?? "https://picsum.photos/seed/diorama/1200/1600"
        return URL(string: raw)!
    }
}
