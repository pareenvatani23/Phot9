/**
 * Minimal Marble smoke-run: image -> public URL (via fal storage) -> Marble world
 * -> write { ok, splatUrl, worldId } (or { ok:false, error }) to <outDir>/marble.json.
 * Never throws / always exits 0 so the workflow can publish the result file and we
 * can read schema-discovery errors cheaply from one tiny artifact.
 *
 *   FAL_KEY=… WORLDLABS_API_KEY=… MARBLE_MODEL=marble-1.0-draft \
 *     node dist/marbleRun.js <imagePath> <outDir>
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { uploadToFal } from "./falClient.js";
import { generateWorldFromImage } from "./marble.js";

const [imgPath, outDir] = process.argv.slice(2);
if (!imgPath || !outDir) {
  console.error("usage: node dist/marbleRun.js <imagePath> <outDir>");
  process.exit(1);
}

await fs.mkdir(outDir, { recursive: true });
const outFile = path.join(outDir, "marble.json");
const model = process.env.MARBLE_MODEL || "marble-1.0-draft";

try {
  // If the image arg is already a public URL, hand it straight to Marble — no
  // fal upload needed (Marble just needs a fetchable URL). Otherwise upload the
  // local file to fal storage to obtain one.
  let imageUrl: string;
  if (/^https?:\/\//i.test(imgPath)) {
    imageUrl = imgPath;
    console.error("Using image URL directly (no fal):", imageUrl);
  } else {
    const image = await fs.readFile(imgPath);
    const isPng = imgPath.toLowerCase().endsWith(".png");
    console.error("Uploading image to fal storage for a public URL…");
    imageUrl = await uploadToFal(image, isPng ? "image/png" : "image/jpeg", `input.${isPng ? "png" : "jpg"}`);
    console.error("image_url:", imageUrl);
  }

  const deadline = Date.now() + 15 * 60 * 1000; // 15 min — world gen can take minutes
  const result = await generateWorldFromImage(imageUrl, { model, deadline, displayName: "diorama" });
  console.error("Marble splatUrl:", result.splatUrl);
  await fs.writeFile(outFile, JSON.stringify({ ok: true, model, splatUrl: result.splatUrl, worldId: result.worldId }, null, 2));
} catch (e) {
  const error = e instanceof Error ? e.message : String(e);
  console.error("Marble run error:", error);
  await fs.writeFile(outFile, JSON.stringify({ ok: false, model, error }, null, 2));
}
