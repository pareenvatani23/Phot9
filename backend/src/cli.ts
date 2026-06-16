/**
 * One-shot pipeline runner (no HTTP). Used by CI on a runner that can reach
 * fal: takes an image path + output dir, runs Stages A–D, and writes
 * `hero.glb` + `backdrop.jpg` + `meta.json` into the output dir.
 *
 *   FAL_KEY=... node dist/cli.js <imagePath> <outDir>
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { buildBackdrop } from "./backdrop.js";
import { uploadToFal, runFal, runFalWithRetry, errMessage } from "./falClient.js";
import { limits } from "./config.js";
import { projectPhotoOntoGLB } from "./texture.js";
import { buildSplat } from "./splat.js";
import type { Sam3BodyOutput } from "./types.js";

/** Pull the first File URL out of an arbitrary fal output shape. */
function extractFileUrl(out: unknown): string | undefined {
  const o = out as Record<string, any>;
  const cand = [o?.image, o?.depth, o?.depth_map, o?.output, o?.images?.[0], o?.image_url];
  for (const c of cand) {
    if (typeof c === "string") return c;
    if (c && typeof c.url === "string") return c.url;
  }
  return undefined;
}

const [imgPath, outDir] = process.argv.slice(2);
if (!imgPath || !outDir) {
  console.error("usage: node dist/cli.js <imagePath> <outDir>");
  process.exit(1);
}

const deadline = Date.now() + limits.PIPELINE_TIMEOUT_MS;

const image = await fs.readFile(imgPath);
await fs.mkdir(outDir, { recursive: true });
const isPng = imgPath.toLowerCase().endsWith(".png");
const contentType = isPng ? "image/png" : "image/jpeg";

console.error("Uploading image to fal…");
const imageUrl = await uploadToFal(image, contentType, `input.${isPng ? "png" : "jpg"}`);
console.error("image_url:", imageUrl);

console.error("Stage A: fal-ai/sam-3/3d-body …");
const body = await runFalWithRetry<Sam3BodyOutput>(
  "fal-ai/sam-3/3d-body",
  { image_url: imageUrl, export_meshes: true, include_3d_keypoints: false },
  { deadline }
);
const people = body.metadata?.people ?? [];
const numPeople = body.metadata?.num_people ?? 0;
console.error("num_people:", numPeople);
if (numPeople <= 0 || people.length === 0) {
  console.error("NO_PEOPLE_DETECTED");
  process.exit(2);
}

console.error("Stage B: backdrop …");
const backdrop = await buildBackdrop(image, people, outDir);

// Texture the RAW camera-space body GLB: projection only maps to the photo in
// the original camera frame, so we deliberately skip 3d-align here (its MoGe
// metric frame would break the projection). Vertex UVs survive any later
// transform, and camera-space geometry also composites naturally with the photo.
console.error("Downloading body GLB …");
const res = await fetch(body.model_glb.url);
if (!res.ok) throw new Error(`body download ${res.status}`);
const bodyGlb = Buffer.from(await res.arrayBuffer());

// Keep prominent (foreground) people; prune small distant passers-by whose
// bbox area is under 15% of the largest detected person.
const areas = people.map((q) => (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]));
const maxArea = Math.max(...areas);
const keepPersons = people.map((_, i) => i).filter((i) => areas[i] >= 0.15 * maxArea);
console.error("people kept:", keepPersons.length, "of", people.length, "(bbox-area filter)");

console.error("Stage C': projecting photo onto mesh …");
const heroPath = path.join(outDir, "hero.glb");
const proj = await projectPhotoOntoGLB(
  bodyGlb,
  image,
  { focalLength: people[0].focal_length, imgW: backdrop.img_w, imgH: backdrop.img_h, keepPersons },
  heroPath
);
console.error("texture convention:", JSON.stringify(proj.convention), "coverage:", proj.coverage.toFixed(3));

// Stage B': best-effort monocular depth for the navigable depth-mesh background.
// Non-fatal — if no model works, the app falls back to a flat backdrop plane.
let depthAvailable = false;
const depthModels = ["fal-ai/imageutils/marigold-depth", "fal-ai/depth-anything/v2"];
for (const m of depthModels) {
  try {
    console.error("Depth: trying", m, "…");
    const out = await runFal<unknown>(m, { image_url: imageUrl }, { deadline });
    const url = extractFileUrl(out);
    if (url) {
      const r = await fetch(url);
      if (r.ok) {
        await fs.writeFile(path.join(outDir, "depth.png"), Buffer.from(await r.arrayBuffer()));
        depthAvailable = true;
        console.error("Depth OK from", m);
        break;
      }
    }
    console.error("Depth: no usable file from", m);
  } catch (e) {
    console.error("Depth model failed", m, errMessage(e));
  }
}

// Stage B'': single-image Gaussian-splat environment from the depth map.
let splatAvailable = false;
if (depthAvailable) {
  try {
    const depthBuf = await fs.readFile(path.join(outDir, "depth.png"));
    const splat = await buildSplat(image, depthBuf, {
      focalLength: people[0].focal_length,
      imgW: backdrop.img_w,
      imgH: backdrop.img_h,
      avgCamTz: people.reduce((s, p) => s + (p.pred_cam_t?.[2] ?? 0), 0) / people.length,
      maxPoints: 500_000,
    });
    await fs.writeFile(path.join(outDir, "scene.splat"), splat);
    splatAvailable = true;
    console.error("Splat OK:", splat.length / 32, "gaussians");
  } catch (e) {
    console.error("Splat build failed:", errMessage(e));
  }
}

const meta = {
  aligned: false,
  textured: true,
  depth_available: depthAvailable,
  splat_available: splatAvailable,
  texture_convention: proj.convention,
  texture_coverage: Number(proj.coverage.toFixed(3)),
  num_people: numPeople,
  img_w: backdrop.img_w,
  img_h: backdrop.img_h,
  avg_cam_tz: people.reduce((s, p) => s + (p.pred_cam_t?.[2] ?? 0), 0) / people.length,
  focal_length: people[0].focal_length,
};
await fs.writeFile(path.join(outDir, "meta.json"), JSON.stringify(meta, null, 2));
console.log(JSON.stringify(meta));
console.error("Done. Wrote hero.glb + backdrop.jpg to", outDir);
