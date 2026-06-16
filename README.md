# Phot9 — Diorama

Turn a group photo into an explorable 3D diorama: the people become real 3D
figures, the landmark behind them becomes a depth-placed backdrop, and you can
orbit the scene (slightly behind the people) instead of staring at a flat photo.
Output is shareable as an auto-orbit video clip.

This repo is a monorepo with both deliverables from the build spec (v1.0):

| dir | what | stack |
|-----|------|-------|
| [`backend/`](backend) | fal pipeline proxy: photo → hero GLB + backdrop | Node 20 / TypeScript / Express |
| [`ios/`](ios) | capture → orbit viewer → MP4 share | Swift / SwiftUI / SceneKit |

## How it works
1. **iOS** captures a still and uploads it to **`POST /v1/diorama`**.
2. **Backend** runs the pipeline on fal.ai:
   - `fal-ai/sam-3/3d-body` — one call reconstructs **all** people into a single combined GLB.
   - backdrop — the original photo becomes a camera-facing plane (no 3D landmark in v1).
   - `fal-ai/sam-3/3d-align` — optional metric alignment, with fallback to the raw body GLB.
3. **iOS** downloads the hero GLB + backdrop and renders a **bounded-orbit** scene
   (azimuth hard-clamped to ±70°), then exports a ~5 s orbit MP4 to the share sheet.

The `FAL_KEY` lives only on the backend and is never shipped in the app.

## Quick start
```bash
# Backend
cd backend && cp .env.example .env   # set FAL_KEY
npm install && npm run dev           # :8080

# iOS
cd ios && xcodegen generate && open Diorama.xcodeproj
# point AppConfig.backendBaseURL (or -DioramaBaseURL) at the backend
```

See [`backend/README.md`](backend/README.md) and [`ios/README.md`](ios/README.md)
for full detail. Known v1 limitations and the v2 backlog are in the build spec §6.
