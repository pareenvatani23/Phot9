import SwiftUI

/// Local assets ready to render in the viewer.
struct ViewerAssets {
    let heroGLB: URL
    let backdrop: UIImage
    let result: DioramaResult
}

/// Top-level flow: capture → processing → viewer (spec §3.2).
struct RootView: View {
    private enum Phase {
        case capture
        case processing(Data)
        case viewer(ViewerAssets)
    }

    @State private var phase: Phase = .capture

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if AppConfig.demoMode {
                DemoView()
            } else {
                content
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        ZStack {
            switch phase {
            case .capture:
                CaptureView { jpeg in
                    phase = .processing(jpeg)
                }
            case .processing(let jpeg):
                ProcessingView(
                    jpeg: jpeg,
                    onReady: { assets in phase = .viewer(assets) },
                    onCancel: { phase = .capture }
                )
            case .viewer(let assets):
                ViewerView(assets: assets) {
                    phase = .capture
                }
            }
        }
    }
}
