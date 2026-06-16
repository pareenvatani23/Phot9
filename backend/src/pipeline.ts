/**
 * The core pipeline (spec 2.4): Stage A reconstruct people -> Stage B backdrop
 * -> Stage C optional metric align -> Stage D package. Runs asynchronously
 * after POST /v1/diorama returns 202; progress is reported via the job store.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { config, limits } from "./config.js";
import { buildBackdrop } from "./backdrop.js";
import { errMessage, runFalWithRetry, uploadToFal } from "./falClient.js";
import { fail, setStage, succeed } from "./jobStore.js";
import { log } from "./logger.js";
import type { DioramaResult, ErrorCode, Sam3AlignOutput, Sam3BodyOutput } from "./types.js";

export const ASSET_ROOT = path.resolve("data", "assets");

class PipelineError extends Error {
  constructor(public code: ErrorCode, message: string) {
    super(message);
  }
}

export async function runPipeline(jobId: string, image: Buffer, contentType: string): Promise<void> {
  const deadline = Date.now() + limits.PIPELINE_TIMEOUT_MS;
  const outDir = path.join(ASSET_ROOT, jobId);

  try {
    // --- Stage: upload original to fal storage --------------------------------
    setStage(jobId, "uploading", 0.1);
    const ext = contentType.includes("png") ? "png" : "jpg";
    const imageUrl = await uploadToFal(image, contentType, `input.${ext}`);

    // --- Stage A: reconstruct all people (one call) ---------------------------
    setStage(jobId, "reconstructing_bodies", 0.4);
    let bodyOut: Sam3BodyOutput;
    try {
      bodyOut = await runFalWithRetry<Sam3BodyOutput>(
        "fal-ai/sam-3/3d-body",
        { image_url: imageUrl, export_meshes: true, include_3d_keypoints: false },
        { deadline }
      );
    } catch (err) {
      if (err instanceof Error && err.message === "TIMEOUT") throw new PipelineError("TIMEOUT", "Pipeline exceeded time budget");
      throw new PipelineError("RECON_FAILED", `Body reconstruction failed: ${errMessage(err)}`);
    }

    const people = bodyOut.metadata?.people ?? [];
    const numPeople = bodyOut.metadata?.num_people ?? 0;
    if (numPeople <= 0 || people.length === 0) {
      throw new PipelineError("NO_PEOPLE_DETECTED", "No people detected in the photo");
    }

    const focalLength = people[0].focal_length;
    const avgCamTz = people.reduce((s, p) => s + (p.pred_cam_t?.[2] ?? 0), 0) / people.length;
    const bboxes = people.map((p) => p.bbox);
    const bodyGlbUrl = bodyOut.model_glb.url;

    // --- Stage B: build the backdrop (CPU only) -------------------------------
    setStage(jobId, "building_backdrop", 0.6);
    const backdrop = await buildBackdrop(image, people, outDir);

    // --- Stage C: align to a metric frame (optional, with fallback) -----------
    setStage(jobId, "aligning", 0.8);
    let aligned = false;
    let heroBuffer: Buffer | null = null;

    try {
      const alignOut = await runFalWithRetry<Sam3AlignOutput>(
        "fal-ai/sam-3/3d-align",
        { image_url: imageUrl, body_mesh_url: bodyGlbUrl, focal_length: focalLength },
        { deadline }
      );
      const scale = alignOut.metadata?.scale_factor;
      const scaleOk = typeof scale === "number" && Number.isFinite(scale) && scale > 0;
      if (scaleOk && alignOut.model_glb?.url) {
        const buf = await download(alignOut.model_glb.url);
        if (buf.length >= limits.MIN_GLB_BYTES) {
          heroBuffer = buf;
          aligned = true;
        } else {
          log.warn("align GLB too small, falling back", { jobId, bytes: buf.length });
        }
      } else {
        log.warn("align returned degenerate scale, falling back", { jobId, scale });
      }
    } catch (err) {
      if (err instanceof Error && err.message === "TIMEOUT") throw new PipelineError("TIMEOUT", "Pipeline exceeded time budget");
      log.warn("align failed, falling back to Stage A GLB", { jobId, error: errMessage(err) });
    }

    if (!heroBuffer) {
      heroBuffer = await download(bodyGlbUrl);
      if (heroBuffer.length < limits.MIN_GLB_BYTES) {
        throw new PipelineError("RECON_FAILED", "Body GLB is empty or unreadable");
      }
    }

    // --- Stage D: package -----------------------------------------------------
    setStage(jobId, "packaging", 0.9);
    const heroPath = path.join(outDir, "hero.glb");
    await fs.writeFile(heroPath, heroBuffer);

    const result: DioramaResult = {
      aligned,
      hero_glb_url: assetUrl(jobId, "hero.glb"),
      backdrop: {
        image_url: assetUrl(jobId, "backdrop.jpg"),
        img_w: backdrop.img_w,
        img_h: backdrop.img_h,
      },
      scene_hint: {
        num_people: numPeople,
        avg_cam_tz: round(avgCamTz, 4),
        focal_length: focalLength,
        people_bboxes: bboxes,
      },
    };

    succeed(jobId, result);
    log.info("job succeeded", { jobId, numPeople, aligned });
  } catch (err) {
    if (err instanceof PipelineError) {
      fail(jobId, { code: err.code, message: err.message });
      log.error("job failed", { jobId, code: err.code, message: err.message });
    } else {
      fail(jobId, { code: "INTERNAL", message: errMessage(err) });
      log.error("job failed (internal)", { jobId, error: errMessage(err) });
    }
  }
}

function assetUrl(jobId: string, name: string): string {
  return `${config.PUBLIC_BASE_URL}/assets/${jobId}/${name}`;
}

async function download(url: string): Promise<Buffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed ${res.status} for ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

function round(n: number, dp: number): number {
  const f = 10 ** dp;
  return Math.round(n * f) / f;
}
