# Diorama — Backend

Thin proxy + pipeline that turns a group photo into the assets the iOS app needs:
a multi-person **hero GLB** and a **backdrop** image. All `fal.ai` calls happen
here so the `FAL_KEY` never ships in the app (spec §0.4, §7).

## Stack
- Node 20+ / TypeScript (ESM)
- `@fal-ai/client` (official client; `@fal-ai/serverless-client` is deprecated)
- `express` + `multer` (multipart upload)
- `sharp` (backdrop + person-mask raster — CPU only)
- In-memory job store (`Map<jobId, JobRecord>`) — no DB for v1

## Setup
```bash
cp .env.example .env   # set FAL_KEY
npm install
npm run dev            # tsx watch, hot reload
# or
npm run build && npm start
```

### Environment
| var | required | default | notes |
|-----|----------|---------|-------|
| `FAL_KEY` | yes | — | fal.ai key, **server-only** |
| `PORT` | no | `8080` | |
| `PUBLIC_BASE_URL` | no | `http://localhost:$PORT` | base for asset URLs the app downloads |

## API (spec §2.3)

### `POST /v1/diorama`
`multipart/form-data`, field `photo` = JPEG/PNG (≤ 12 MB). Creates a job, starts
the pipeline, returns immediately.
```json
202 { "job_id": "uuid", "status": "queued" }
```

### `GET /v1/diorama/{job_id}`
```json
{ "job_id": "uuid", "status": "running", "stage": "reconstructing_bodies", "progress": 0.4 }
{ "job_id": "uuid", "status": "succeeded", "result": { ... } }   // see Result below
{ "job_id": "uuid", "status": "failed", "error": { "code": "NO_PEOPLE_DETECTED", "message": "..." } }
```
`status` ∈ `queued | running | succeeded | failed`.
`stage` ∈ `uploading | segmenting | reconstructing_bodies | building_backdrop | aligning | packaging`.

### Assets
`GET /assets/{job_id}/hero.glb` and `/assets/{job_id}/backdrop.jpg` — served from
`./data/assets`, re-hosted off fal's expiring URLs.

## Result schema (spec §2.6)
```json
{
  "aligned": true,
  "hero_glb_url": "https://<PUBLIC_BASE_URL>/assets/<job>/hero.glb",
  "backdrop": { "image_url": "https://<PUBLIC_BASE_URL>/assets/<job>/backdrop.jpg", "img_w": 4032, "img_h": 3024 },
  "scene_hint": { "num_people": 4, "avg_cam_tz": 5.8, "focal_length": 1100.0, "people_bboxes": [[x0,y0,x1,y1]] }
}
```

## Pipeline (spec §2.4)
1. **Upload** original to fal storage → `image_url`.
2. **Stage A** `fal-ai/sam-3/3d-body` (one call, all people) → combined GLB + per-person metadata. `num_people == 0` ⇒ fail `NO_PEOPLE_DETECTED`.
3. **Stage B** backdrop: re-encode original to `backdrop.jpg`; rasterize the union of person bboxes (each expanded 8%) to `people_mask.png` (kept for the v2 inpainting step). CPU only, $0.
4. **Stage C** `fal-ai/sam-3/3d-align` (optional). On degenerate result (`scale_factor ≤ 0` / non-finite / GLB < 1 KB) or any error, **fall back** to the Stage A GLB. `result.aligned` records which was used.
5. **Stage D** download the chosen hero GLB + backdrop, re-host under `PUBLIC_BASE_URL`, assemble `result`, mark `succeeded`.

Failure handling: fal errors retry once then fail `RECON_FAILED`; total pipeline > 180 s fails `TIMEOUT`. Poll interval 2 s.

## Cost (reference)
`3d-body` $0.02 + `3d-align` $0.02 (skipped on fallback) + backdrop $0 ≈ **$0.04 / diorama**, regardless of group size.

## Scripts
- `npm run dev` — hot-reload dev server
- `npm run build` / `npm start` — compile then run
- `npm run typecheck` — `tsc --noEmit`
