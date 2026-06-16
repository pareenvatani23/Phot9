/**
 * Single-image Gaussian-splat environment generator.
 *
 * Back-projects every pixel of the photo to 3D using the monocular depth map
 * and the same pinhole intrinsics as the people meshes, emitting one Gaussian
 * per (sub-sampled) pixel in the compact `.splat` format (antimatter15):
 *   position: 3×f32 | scale: 3×f32 | color: 4×u8 RGBA | rotation: 4×u8 (w,x,y,z)
 *
 * This is the depth→point-cloud→splat approach (2.5D: no occluded geometry, so
 * it stretches when viewed far off-axis). The result lives in the SAME camera
 * space as the textured people, so they composite directly.
 */
import sharp from "sharp";

export interface SplatParams {
  focalLength: number;
  imgW: number;
  imgH: number;
  avgCamTz: number;
  /** cap on Gaussian count (sub-samples to fit) */
  maxPoints?: number;
  /** set true if the depth map encodes far=bright (we default to near=bright) */
  invertDepth?: boolean;
}

const SH_C0 = 0.28209479177387814;

export async function buildSplat(backdropJpeg: Buffer, depthPng: Buffer, p: SplatParams): Promise<Buffer> {
  const W = p.imgW, H = p.imgH;

  // RGB photo at WxH.
  const rgb = await sharp(backdropJpeg).removeAlpha().resize(W, H, { fit: "fill" })
    .raw().toBuffer(); // W*H*3 u8

  // 16-bit depth at WxH, as raw little-endian ushort.
  const depthRaw = await sharp(depthPng).resize(W, H, { fit: "fill" })
    .toColourspace("b-w").raw({ depth: "ushort" }).toBuffer(); // W*H*2 bytes
  const depth = new Uint16Array(depthRaw.buffer, depthRaw.byteOffset, W * H);

  const cx = W / 2, cy = H / 2, f = p.focalLength;
  // Compressed depth span keeps the cloud denser and avoids extreme stretching
  // of far points (foreground ~ people depth, background ~3× back).
  const zNear = Math.max(0.3, p.avgCamTz * 0.75);
  const zFar = Math.max(zNear + 1, p.avgCamTz * 3.0);

  // Sub-sample to respect maxPoints.
  const maxPoints = p.maxPoints ?? 500_000;
  const stride = Math.max(1, Math.round(Math.sqrt((W * H) / maxPoints)));
  const cols = Math.floor(W / stride), rows = Math.floor(H / stride);
  const count = cols * rows;

  const out = Buffer.alloc(count * 32);
  let o = 0;
  for (let j = 0; j < rows; j++) {
    const py = j * stride;
    for (let i = 0; i < cols; i++) {
      const px = i * stride;
      const di = py * W + px;
      let norm = depth[di] / 65535; // 0..1
      if (p.invertDepth) norm = 1 - norm; // make 1 = near
      const Z = zFar - norm * (zFar - zNear);

      // Back-project (pinhole; camera looks down -Z, Y up).
      const x = ((px - cx) / f) * Z;
      const y = -((py - cy) / f) * Z;
      const z = -Z;

      const ci = di * 3;
      const r = rgb[ci], g = rgb[ci + 1], b = rgb[ci + 2];

      // Gaussian footprint ≈ one pixel at depth Z, slightly enlarged to overlap.
      const s = (Z / f) * stride * 1.6;

      out.writeFloatLE(x, o); out.writeFloatLE(y, o + 4); out.writeFloatLE(z, o + 8);
      out.writeFloatLE(s, o + 12); out.writeFloatLE(s, o + 16); out.writeFloatLE(s, o + 20);
      out[o + 24] = r; out[o + 25] = g; out[o + 26] = b; out[o + 27] = 255;
      // identity rotation (w,x,y,z) = (1,0,0,0), packed as q*128+128
      out[o + 28] = 255; out[o + 29] = 128; out[o + 30] = 128; out[o + 31] = 128;
      o += 32;
    }
  }
  return out;
}

/** Convenience: keep SH constant handy for a future .ply exporter. */
export const SH_DC = SH_C0;
