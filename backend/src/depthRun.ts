/**
 * Depth 3D-photo runner: single image -> monocular depth map -> a manifest the
 * depth.html viewer lifts into one camera-consistent 3D relief (person + scene
 * share the photo's geometry, so perspective/positions are correct by design).
 *
 * Writes <outDir>/depth.local.json { ok, photoUrl, depthUrl, manifestUrl, viewerUrl }.
 *
 *   FAL_KEY=… node dist/depthRun.js <imagePath> <outDir>
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import sharp from "sharp";
import { runFal, uploadToFal } from "./falClient.js";

const VIEWER = "https://pareenvatani23.github.io/Phot9/depth.html";
const [imgPath, outDir] = process.argv.slice(2);
if (!imgPath || !outDir) { console.error("usage: node dist/depthRun.js <imagePath> <outDir>"); process.exit(1); }
await fs.mkdir(outDir, { recursive: true });
const outFile = path.join(outDir, "depth.local.json");
const deadline = Date.now() + 8 * 60 * 1000;

function extractImageUrl(o: any): string | undefined {
  return o?.image?.url ?? o?.depth?.url ?? o?.images?.[0]?.url
    ?? (typeof o === "object" ? Object.values(o).flatMap((v: any) => (v && v.url ? [v.url] : []))[0] : undefined);
}

try {
  const image = await fs.readFile(imgPath);
  const meta = await sharp(image).metadata();
  const W = meta.width ?? 0, H = meta.height ?? 0;
  if (!W || !H) throw new Error("could not read image dimensions");
  const isPng = imgPath.toLowerCase().endsWith(".png");

  console.error("Uploading photo to fal…");
  const photoUrl = await uploadToFal(image, isPng ? "image/png" : "image/jpeg", `photo.${isPng ? "png" : "jpg"}`);

  console.error("Depth Anything v2…");
  const dep = await runFal<any>("fal-ai/image-preprocessors/depth-anything/v2", { image_url: photoUrl }, { deadline });
  const depthUrl = extractImageUrl(dep);
  if (!depthUrl) throw new Error("no depth image in response: " + JSON.stringify(dep).slice(0, 300));

  const manifest = { photo: { url: photoUrl, w: W, h: H }, depthUrl };
  const manifestUrl = await uploadToFal(Buffer.from(JSON.stringify(manifest)), "application/json", "depth.json");
  const viewerUrl = `${VIEWER}?depth=${encodeURIComponent(manifestUrl)}`;

  await fs.writeFile(outFile, JSON.stringify({ ok: true, photoUrl, depthUrl, manifestUrl, viewerUrl }, null, 2));
  console.error("VIEWER:", viewerUrl);
} catch (e) {
  const error = e instanceof Error ? e.message : String(e);
  console.error("depth run error:", error);
  await fs.writeFile(outFile, JSON.stringify({ ok: false, error }, null, 2));
}
