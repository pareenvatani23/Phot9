# Diorama — iOS app

Captures a group photo, sends it to the backend, and renders the returned hero
GLB + backdrop as an explorable, bounded-orbit 3D scene with a one-tap orbit
video export (spec §3).

## Stack
- iOS 16+, SwiftUI + SceneKit, Swift 5.9
- **GLTFKit2** (SPM) — loads GLB into SceneKit (`SCNScene(url:)` cannot read GLB)
- Vision (`VNDetectHumanRectanglesRequest`) for the live framing overlay
- AVFoundation for capture + `AVAssetWriter` MP4 export

## Generate the project
The `.xcodeproj` is generated (not committed) via [XcodeGen](https://github.com/yonaskolb/XcodeGen):
```bash
brew install xcodegen
cd ios
xcodegen generate
open Diorama.xcodeproj
```
`project.yml` declares the GLTFKit2 SPM package (pin it to a release tag for
production) and the camera/photo-library usage strings.

## Configure the backend URL
`AppConfig.backendBaseURL` defaults to `http://localhost:8080`. Override without
recompiling via a launch argument / user default key `DioramaBaseURL`, e.g. set
it in the scheme's arguments: `-DioramaBaseURL https://your-backend.example`.

> The `FAL_KEY` is **never** in the app — the app only talks to the backend (spec §7).

## Structure
```
Diorama/
  App/         DioramaApp.swift, AppConfig.swift
  Models/      APIModels.swift            (Codable mirrors of the API contracts §2.6)
  Networking/  DioramaAPI.swift           (multipart upload, 2 s poll, asset download)
  Capture/     CameraController.swift     (AVCaptureSession + Vision overlay)
  Scene/       DioramaSceneController.swift (scene assembly + bounded orbit §3.3/§3.4)
               OrbitRecorder.swift        (auto-orbit MP4 export §3.5)
  Views/       RootView, CaptureView, ProcessingView, ViewerView
  Resources/   Info.plist
```

## The bounded orbit (the core interaction, §3.4)
`allowsCameraControl` is OFF. A custom orbit recomputes the camera from
`(azimuth, elevation, orbitRadius)` each gesture, with the azimuth **hard
clamped to ±70°** (the "void guard") and rubber-band resistance near the edge,
so the user gets strong parallax without swinging to the unphotographed rear.

## Orbit video (§3.5)
"Record orbit" sweeps azimuth `0 → −0.9 → +0.9 → 0` over ~5 s (ease-in-out),
snapshots each frame, and writes H.264 30 fps 1080×1920 via `AVAssetWriter`,
then presents the system share sheet.
