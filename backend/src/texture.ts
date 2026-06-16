/**
 * Project the original photo onto the reconstructed body mesh (the "real body
 * texture" pass). SAM 3D Body returns the combined mesh already positioned in
 * camera space, so we can re-project each vertex back into the image with a
 * pinhole camera (focal length + image-centre principal point) and use that as
 * its UV. The front of each person then samples real skin/clothing; back-facing
 * vertices smear, which is why the viewer clamps the orbit to a forward arc.
 *
 * NOTE: the exact sign convention (Y up/down, Z sign) of the SAM camera is not
 * documented, so `flipY` / `flipZ` are tunable and calibrated against the first
 * render. We pick the convention that lands the most vertices inside the image.
 */
import { Document, NodeIO } from "@gltf-transform/core";
import sharp from "sharp";

export interface ProjectParams {
  focalLength: number;
  imgW: number;
  imgH: number;
  /** principal point; defaults to image centre */
  cx?: number;
  cy?: number;
  /** person node indices to keep (others, e.g. distant passers-by, are pruned) */
  keepPersons?: number[];
}

interface Convention {
  flipY: boolean;
  flipZ: boolean;
}

/** Project a single vertex (camera space) to normalized UV under a convention. */
function projectUV(
  x: number,
  y: number,
  z: number,
  p: Required<ProjectParams>,
  c: Convention
): [number, number] | null {
  const zz = c.flipZ ? -z : z;
  if (zz <= 1e-6) return null; // behind camera
  const yy = c.flipY ? -y : y;
  const u = (p.focalLength * x) / zz + p.cx;
  const v = (p.focalLength * yy) / zz + p.cy;
  return [u / p.imgW, v / p.imgH];
}

/** Fraction of vertices whose projection lands inside the image, for calibration. */
function inFrameScore(positions: Float32Array, p: Required<ProjectParams>, c: Convention): number {
  let inside = 0;
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    const uv = projectUV(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2], p, c);
    if (uv && uv[0] >= 0 && uv[0] <= 1 && uv[1] >= 0 && uv[1] <= 1) inside++;
  }
  return n === 0 ? 0 : inside / n;
}

/**
 * Read `inGlb`, add projected UVs + the photo as baseColorTexture to every
 * primitive, and write `outGlb`. Returns the chosen convention + coverage.
 */
export async function projectPhotoOntoGLB(
  inGlb: Buffer,
  photoJpeg: Buffer,
  params: ProjectParams,
  outPath: string
): Promise<{ convention: Convention; coverage: number }> {
  const io = new NodeIO();
  const doc = await io.readBinary(new Uint8Array(inGlb));
  const p: Required<ProjectParams> = {
    ...params,
    cx: params.cx ?? params.imgW / 2,
    cy: params.cy ?? params.imgH / 2,
    keepPersons: params.keepPersons ?? [],
  };

  // Prune background "person_NN" nodes not in the keep set (distant passers-by
  // stay in the photo/point-cloud backdrop, but not as floating 3D meshes).
  if (p.keepPersons.length > 0) {
    const keep = new Set(p.keepPersons);
    for (const node of doc.getRoot().listNodes()) {
      const m = /^person_(\d+)$/.exec(node.getName() ?? "");
      if (m && !keep.has(parseInt(m[1], 10))) {
        node.getMesh()?.dispose();
        node.dispose();
      }
    }
  }

  // Gather all vertex positions to pick the best sign convention.
  const meshes = doc.getRoot().listMeshes();
  const allPos: number[] = [];
  for (const mesh of meshes) {
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute("POSITION");
      if (pos) allPos.push(...pos.getArray()!);
    }
  }
  const posArr = new Float32Array(allPos);
  const candidates: Convention[] = [
    { flipY: true, flipZ: false },
    { flipY: false, flipZ: false },
    { flipY: true, flipZ: true },
    { flipY: false, flipZ: true },
  ];
  let best = candidates[0];
  let bestScore = -1;
  for (const c of candidates) {
    const s = inFrameScore(posArr, p, c);
    if (s > bestScore) {
      bestScore = s;
      best = c;
    }
  }

  // Encode the photo as PNG for the texture (broad GLB viewer support).
  const png = await sharp(photoJpeg).png().toBuffer();
  const texture = doc.createTexture("photo").setImage(new Uint8Array(png)).setMimeType("image/png");
  const material = doc
    .createMaterial("projected")
    .setBaseColorTexture(texture)
    .setRoughnessFactor(0.9)
    .setMetallicFactor(0.0);
  material.getBaseColorTextureInfo()?.setTexCoord(0);

  for (const mesh of meshes) {
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute("POSITION");
      if (!pos) continue;
      const arr = pos.getArray()!;
      const n = arr.length / 3;
      const uvs = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) {
        const uv = projectUV(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2], p, best);
        // glTF UV origin is top-left; image v already top-left from our formula.
        uvs[i * 2] = uv ? Math.min(1, Math.max(0, uv[0])) : 0;
        uvs[i * 2 + 1] = uv ? Math.min(1, Math.max(0, uv[1])) : 0;
      }
      const accessor = doc.createAccessor().setType("VEC2").setArray(uvs);
      prim.setAttribute("TEXCOORD_0", accessor);
      prim.setMaterial(material);
    }
  }

  const out = await io.writeBinary(doc);
  const { promises: fs } = await import("node:fs");
  await fs.writeFile(outPath, Buffer.from(out));
  return { convention: best, coverage: bestScore };
}
