/**
 * Minimal Marble smoke-run: image -> public URL (via fal storage) -> Marble world
 * -> write { splatUrl, worldId } to <outDir>/marble.json. Isolated from the heavy
 * SAM/depth pipeline so we can validate World Labs cheaply (use the draft model).
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

const image = await fs.readFile(imgPath);
await fs.mkdir(outDir, { recursive: true });
const isPng = imgPath.toLowerCase().endsWith(".png");

console.error("Uploading image to fal storage for a public URL…");
const imageUrl = await uploadToFal(image, isPng ? "image/png" : "image/jpeg", `input.${isPng ? "png" : "jpg"}`);
console.error("image_url:", imageUrl);

const model = process.env.MARBLE_MODEL || "marble-1.0-draft";
const deadline = Date.now() + 15 * 60 * 1000; // 15 min — world gen can take minutes
const result = await generateWorldFromImage(imageUrl, { model, deadline, displayName: "diorama" });

console.error("Marble splatUrl:", result.splatUrl);
await fs.writeFile(
  path.join(outDir, "marble.json"),
  JSON.stringify({ splatUrl: result.splatUrl, worldId: result.worldId, model }, null, 2)
);
console.error("Wrote", path.join(outDir, "marble.json"));
