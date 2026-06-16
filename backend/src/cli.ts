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
import { uploadToFal, runFalWithRetry, errMessage } from "./falClient.js";
import { limits } from "./config.js";
import type { Sam3AlignOutput, Sam3BodyOutput } from "./types.js";

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

console.error("Stage C: fal-ai/sam-3/3d-align …");
let heroUrl = body.model_glb.url;
let aligned = false;
try {
  const align = await runFalWithRetry<Sam3AlignOutput>(
    "fal-ai/sam-3/3d-align",
    { image_url: imageUrl, body_mesh_url: body.model_glb.url, focal_length: people[0].focal_length },
    { deadline }
  );
  const s = align.metadata?.scale_factor;
  if (typeof s === "number" && Number.isFinite(s) && s > 0 && align.model_glb?.url) {
    heroUrl = align.model_glb.url;
    aligned = true;
  } else {
    console.error("align degenerate; using Stage A GLB");
  }
} catch (e) {
  console.error("align failed; using Stage A GLB:", errMessage(e));
}

console.error("Downloading hero GLB …");
const res = await fetch(heroUrl);
if (!res.ok) throw new Error(`hero download ${res.status}`);
await fs.writeFile(path.join(outDir, "hero.glb"), Buffer.from(await res.arrayBuffer()));

const meta = {
  aligned,
  num_people: numPeople,
  img_w: backdrop.img_w,
  img_h: backdrop.img_h,
  avg_cam_tz: people.reduce((s, p) => s + (p.pred_cam_t?.[2] ?? 0), 0) / people.length,
  focal_length: people[0].focal_length,
};
await fs.writeFile(path.join(outDir, "meta.json"), JSON.stringify(meta, null, 2));
console.log(JSON.stringify(meta));
console.error("Done. Wrote hero.glb + backdrop.jpg to", outDir);
