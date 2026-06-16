/**
 * Stage B (spec 2.4): build the backdrop. The landmark/background is NOT
 * reconstructed as a 3D mesh — it becomes a camera-facing textured plane.
 *
 * v1 backdrop is the unmodified original image. The people pixels remain in the
 * backdrop but are hidden behind the 3D people meshes within the bounded orbit
 * arc (see spec section 6 — inpainting the people out is a v2 improvement).
 *
 * We also rasterize the person-region union mask (bboxes expanded 8%). It is
 * not referenced in the v1 result (people_alpha is optional), but it is the
 * exact artifact the v2 inpainting step consumes, so we produce + persist it.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import sharp from "sharp";
import { limits } from "./config.js";
import type { Sam3BodyPerson } from "./types.js";

export interface BackdropResult {
  img_w: number;
  img_h: number;
  backdropPath: string; // absolute path to backdrop.jpg
  maskPath: string; // absolute path to people_mask.png (for v2)
}

/** Expand a pixel bbox by `frac` of its own size on every side, clamped to image. */
export function expandBbox(
  bbox: [number, number, number, number],
  imgW: number,
  imgH: number,
  frac: number
): [number, number, number, number] {
  const [x0, y0, x1, y1] = bbox;
  const w = x1 - x0;
  const h = y1 - y0;
  const dx = w * frac;
  const dy = h * frac;
  return [
    Math.max(0, Math.floor(x0 - dx)),
    Math.max(0, Math.floor(y0 - dy)),
    Math.min(imgW, Math.ceil(x1 + dx)),
    Math.min(imgH, Math.ceil(y1 + dy)),
  ];
}

export async function buildBackdrop(
  originalImage: Buffer,
  people: Sam3BodyPerson[],
  outDir: string
): Promise<BackdropResult> {
  await fs.mkdir(outDir, { recursive: true });

  const meta = await sharp(originalImage).metadata();
  const img_w = meta.width ?? 0;
  const img_h = meta.height ?? 0;
  if (img_w <= 0 || img_h <= 0) {
    throw new Error("Could not read image dimensions for backdrop");
  }

  // backdrop.jpg — re-encode the original to a consistent JPEG.
  const backdropPath = path.join(outDir, "backdrop.jpg");
  await sharp(originalImage).jpeg({ quality: 92 }).toFile(backdropPath);

  // people_mask.png — union of person bboxes (each expanded 8%) as white on black.
  const rects = people
    .map((p) => expandBbox(p.bbox, img_w, img_h, limits.BBOX_EXPAND))
    .map(([x0, y0, x1, y1]) => `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="#ffffff"/>`)
    .join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${img_w}" height="${img_h}"><rect width="100%" height="100%" fill="#000000"/>${rects}</svg>`;
  const maskPath = path.join(outDir, "people_mask.png");
  await sharp(Buffer.from(svg)).png().toFile(maskPath);

  return { img_w, img_h, backdropPath, maskPath };
}
