/**
 * Re-mesh experiment: take an EXISTING scene.json (its world + person cutouts are
 * reused as-is) and regenerate just the per-person 3D meshes with a different
 * image-to-3D model, so we can A/B model quality cheaply (no re-matte, no Marble).
 *
 * Writes <outDir>/remesh.local.json { ok, model, sceneUrl, viewerUrl, meshed }.
 * Never throws; per-person failures fall back to that person's billboard.
 *
 *   FAL_KEY=… MESH_MODEL=rodin node dist/remeshRun.js <baseSceneUrl> <outDir>
 *   MESH_MODEL ∈ rodin | tripo | hunyuan3d-v3 | hunyuan3d-v2
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { runFal, uploadToFal } from "./falClient.js";

const VIEWER = "https://pareenvatani23.github.io/Phot9/";
const [baseSceneUrl, outDir] = process.argv.slice(2);
if (!baseSceneUrl || !outDir) {
  console.error("usage: node dist/remeshRun.js <baseSceneUrl> <outDir>");
  process.exit(1);
}
await fs.mkdir(outDir, { recursive: true });
const meshModel = process.env.MESH_MODEL || "rodin";
const outFile = path.join(outDir, `remesh_${meshModel}.json`);
const deadline = Date.now() + 25 * 60 * 1000;

/** Recursively find a .glb URL (model_mesh.url / model_glb.url / any *.glb). */
function extractGlbUrl(obj: unknown): string | undefined {
  const hits: string[] = [];
  const walk = (v: unknown) => {
    if (typeof v === "string") {
      if (/^https?:\/\/\S+\.glb(\?|$)/i.test(v)) hits.push(v);
    } else if (Array.isArray(v)) v.forEach(walk);
    else if (v && typeof v === "object") Object.values(v as Record<string, unknown>).forEach(walk);
  };
  walk(obj);
  return hits[0];
}

/** Map a model name to its fal endpoint + input shape for a single cutout URL. */
function meshCall(model: string, cutoutUrl: string): { endpoint: string; input: Record<string, unknown> } {
  switch (model) {
    case "rodin":
      return { endpoint: "fal-ai/hyper3d/rodin", input: { input_image_urls: [cutoutUrl], geometry_file_format: "glb", material: "PBR" } };
    case "tripo":
      return { endpoint: "tripo3d/tripo/v2.5/image-to-3d", input: { image_url: cutoutUrl, texture: true, pbr: true } };
    case "sam3body": // human-specialist: anatomy-correct body mesh (keeps pose)
      return { endpoint: "fal-ai/sam-3/3d-body", input: { image_url: cutoutUrl, export_meshes: true } };
    case "hunyuan3d-v3":
      return { endpoint: "fal-ai/hunyuan3d-v3/image-to-3d", input: { input_image_url: cutoutUrl, textured_mesh: true } };
    case "hunyuan3d-v2":
    default:
      return { endpoint: "fal-ai/hunyuan3d/v2", input: { input_image_url: cutoutUrl, textured_mesh: true, octree_resolution: 256 } };
  }
}

try {
  const base: any = await (await fetch(baseSceneUrl)).json();
  const people: any[] = base.people ?? [];
  if (people.length === 0) throw new Error("base scene has no people");

  let meshed = 0;
  for (const p of people) {
    try {
      const { endpoint, input } = meshCall(meshModel, p.cutoutUrl);
      console.error(`${meshModel}: meshing person ${p.id} via ${endpoint}…`);
      const out = await runFal<any>(endpoint, input, { deadline });
      const url = out?.model_mesh?.url ?? extractGlbUrl(out);
      if (url) { p.meshUrl = url; meshed++; }
      else console.error(`person ${p.id}: no glb in response`);
    } catch (e) {
      p.meshUrl = undefined; // keep billboard
      console.error(`person ${p.id} failed (${meshModel}):`, e instanceof Error ? e.message : e);
    }
  }

  const scene = { world: base.world, photo: base.photo, people, meshModel };
  const sceneUrl = await uploadToFal(Buffer.from(JSON.stringify(scene)), "application/json", `scene_${meshModel}.json`);
  const viewerUrl = `${VIEWER}?scene=${encodeURIComponent(sceneUrl)}`;
  await fs.writeFile(outFile, JSON.stringify({ ok: true, model: meshModel, meshed, sceneUrl, viewerUrl }, null, 2));
  console.error("VIEWER:", viewerUrl);
} catch (e) {
  const error = e instanceof Error ? e.message : String(e);
  console.error("remesh error:", error);
  await fs.writeFile(outFile, JSON.stringify({ ok: false, model: meshModel, error }, null, 2));
}
