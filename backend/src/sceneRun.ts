/**
 * Hybrid scene builder: group photo -> crisp people cut out as billboards +
 * a clean Marble world with the people erased. Writes a scene.json manifest the
 * web viewer renders via ?scene=<url>.
 *
 * Steps:
 *   1. upload photo to fal storage (public URL)
 *   2. fal-ai/sam-3/3d-body  -> per-person bboxes (multi-person detector)
 *   3. per person: crop -> fal-ai/birefnet/v2 -> RGBA cutout (the billboard)
 *   4. union the cutout alphas -> people mask -> fal-ai/bria/eraser -> clean plate
 *   5. Marble on the clean plate -> ghost-free splat world
 *   6. emit scene.json { world, photo{w,h}, people[{cutoutUrl, bbox}] } to fal
 *
 * Never throws: always writes <outDir>/scene.local.json ({ok:true|false,...})
 * and exits 0 so CI can publish the result + we can read URLs from one artifact.
 *
 *   FAL_KEY=… WORLDLABS_API_KEY=… MARBLE_MODEL=marble-1.1 \
 *     node dist/sceneRun.js <imagePath> <outDir>
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import sharp from "sharp";
import { runFal, uploadToFal } from "./falClient.js";
import { expandBbox } from "./backdrop.js";
import { generateWorldFromImage } from "./marble.js";
import type { Sam3BodyOutput } from "./types.js";

const VIEWER = "https://pareenvatani23.github.io/Phot9/";
const MAX_PEOPLE = 12;
const BBOX_EXPAND = 0.08; // matches the union-mask margin used elsewhere

interface FalImageOut { image: { url: string; width?: number; height?: number } }

const [imgPath, outDir] = process.argv.slice(2);
if (!imgPath || !outDir) {
  console.error("usage: node dist/sceneRun.js <imagePath> <outDir>");
  process.exit(1);
}
await fs.mkdir(outDir, { recursive: true });
const outFile = path.join(outDir, "scene.local.json");
const model = process.env.MARBLE_MODEL || "marble-1.1";
const deadline = Date.now() + 20 * 60 * 1000;

async function download(url: string): Promise<Buffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download ${res.status} for ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

try {
  const image = await fs.readFile(imgPath);
  const meta = await sharp(image).metadata();
  const W = meta.width ?? 0, H = meta.height ?? 0;
  if (!W || !H) throw new Error("could not read image dimensions");

  const isPng = imgPath.toLowerCase().endsWith(".png");
  console.error("Uploading photo to fal…");
  const imageUrl = await uploadToFal(image, isPng ? "image/png" : "image/jpeg", `input.${isPng ? "png" : "jpg"}`);

  // ── 2. detect people ──────────────────────────────────────────────────────
  console.error("SAM-3: detecting people…");
  const body = await runFal<Sam3BodyOutput>(
    "fal-ai/sam-3/3d-body",
    { image_url: imageUrl, export_meshes: false, include_3d_keypoints: false },
    { deadline }
  );
  const detected = (body.metadata?.people ?? [])
    .map((p) => p.bbox)
    .filter((b) => Array.isArray(b) && b.length === 4)
    .sort((a, b) => (b[2] - b[0]) * (b[3] - b[1]) - (a[2] - a[0]) * (a[3] - a[1]))
    .slice(0, MAX_PEOPLE);
  console.error(`SAM-3: ${detected.length} people`);
  if (detected.length === 0) throw new Error("no people detected");

  // ── 3. per-person cutouts (BiRefNet on the crop) ──────────────────────────
  const people: { id: number; cutoutUrl: string; bbox: [number, number, number, number] }[] = [];
  const maskTiles: { input: Buffer; left: number; top: number }[] = [];

  for (let i = 0; i < detected.length; i++) {
    const crop = expandBbox(detected[i] as [number, number, number, number], W, H, BBOX_EXPAND);
    const [cx0, cy0, cx1, cy1] = crop;
    const cw = cx1 - cx0, ch = cy1 - cy0;
    if (cw < 8 || ch < 8) continue;

    const cropBuf = await sharp(image).extract({ left: cx0, top: cy0, width: cw, height: ch }).png().toBuffer();
    const cropUrl = await uploadToFal(cropBuf, "image/png", `crop${i}.png`);

    console.error(`BiRefNet: matting person ${i}…`);
    const bf = await runFal<FalImageOut>(
      "fal-ai/birefnet/v2",
      { image_url: cropUrl, model: "Portrait", output_format: "png", refine_foreground: true },
      { deadline }
    );

    // Normalise the cutout back to the exact crop rect so it aligns 1:1 with bbox.
    const rgba = await sharp(await download(bf.image.url)).resize(cw, ch, { fit: "fill" }).png().toBuffer();
    const cutoutUrl = await uploadToFal(rgba, "image/png", `person${i}.png`);
    people.push({ id: i, cutoutUrl, bbox: crop });

    // White RGB + the person's alpha -> a tile that paints a white silhouette
    // when composited (over black) into the union mask.
    const alphaPng = await sharp(rgba).ensureAlpha().extractChannel(3).png().toBuffer();
    const tile = await sharp({ create: { width: cw, height: ch, channels: 3, background: "#ffffff" } })
      .joinChannel(alphaPng).png().toBuffer();
    maskTiles.push({ input: tile, left: cx0, top: cy0 });
  }
  if (people.length === 0) throw new Error("no usable person crops");

  // ── 4. union mask -> erase people -> clean plate ──────────────────────────
  console.error("Building people mask + erasing…");
  let unionMask = await sharp({ create: { width: W, height: H, channels: 3, background: "#000000" } })
    .composite(maskTiles).png().toBuffer();
  // light dilation so the eraser covers edges/halos
  unionMask = await sharp(unionMask).blur(4).threshold(30).png().toBuffer();
  const maskUrl = await uploadToFal(unionMask, "image/png", "people_mask.png");

  const eraser = await runFal<FalImageOut>(
    "fal-ai/bria/eraser",
    { image_url: imageUrl, mask_url: maskUrl },
    { deadline }
  );
  const cleanUrl = eraser.image.url;
  console.error("clean plate:", cleanUrl);

  // ── 5. Marble world on the clean plate ────────────────────────────────────
  console.error(`Marble (${model}) on clean plate…`);
  const marble = await generateWorldFromImage(cleanUrl, { model, deadline, displayName: "diorama-scene" });
  console.error("splatUrl:", marble.splatUrl);

  // ── 6. emit scene.json ────────────────────────────────────────────────────
  const scene = {
    world: { splatUrl: marble.splatUrl, model, worldId: marble.worldId },
    photo: { w: W, h: H },
    people: people.map((p) => ({ id: p.id, cutoutUrl: p.cutoutUrl, bbox: p.bbox })),
  };
  const sceneUrl = await uploadToFal(Buffer.from(JSON.stringify(scene)), "application/json", "scene.json");
  const viewerUrl = `${VIEWER}?scene=${encodeURIComponent(sceneUrl)}`;

  await fs.writeFile(outFile, JSON.stringify(
    { ok: true, model, numPeople: people.length, sceneUrl, viewerUrl, cleanUrl, scene }, null, 2));
  console.error("VIEWER:", viewerUrl);
} catch (e) {
  const error = e instanceof Error ? e.message : String(e);
  console.error("scene run error:", error);
  await fs.writeFile(outFile, JSON.stringify({ ok: false, model, error }, null, 2));
}
