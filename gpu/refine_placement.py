"""Differentiable-render refinement of the hero placement (step (b), on Modal GPU).

The geometric solve in web/auto.html gives a good initial rigid transform (foot
position from a ground raycast, metric scale, face-camera yaw). This refines it by
analysis-by-synthesis: render the hero's silhouette with PyTorch3D from the capture
camera and optimise (translation, yaw, uniform scale) so the rendered silhouette
matches the person mask from the original photo, while keeping the feet on the
scene ground and out of the geometry.

This is the PHOSA / PROX / MOVER recipe specialised to a rigid placement:
  loss = silhouette + ground-contact + non-penetration + scale prior.

Inputs (all passed to `refine.remote(...)`):
  hero_obj_bytes : PSHuman .obj (vertex-coloured); normalised here like the viewer.
  mask_png_bytes : person segmentation mask from the ORIGINAL photo (white = person).
  ground_ply_bytes : world_ground.ply (sky removed) — for the ground plane + collision.
  init : {x,y,z,yaw_deg,scale} from auto.html's solveGround.
  fov_deg : vertical FOV of the capture camera (HunyuanWorld panorama front ≈ viewer's 70°).

Returns the refined {x,y,z,yaw_deg,scale} — drop straight into auto.html as
?hx&hy&hz&hr&hs. Run offline; an A100 is overkill but the image is shared with the
other GPU steps.
"""
from __future__ import annotations

import modal

app = modal.App("phot9-refine-placement")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .env({"CC": "gcc", "CXX": "g++", "TORCH_CUDA_ARCH_LIST": "8.0",
          "FORCE_CUDA": "1", "MAX_JOBS": "4"})
    .pip_install("torch==2.5.0", "torchvision==0.20.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.24.1", "pillow", "trimesh")
    # build deps for pytorch3d's --no-build-isolation source build (else the
    # legacy setup.py path dies with "invalid command 'bdist_wheel'").
    .pip_install("setuptools", "wheel", "ninja")
    .run_commands("pip install --no-build-isolation "
                  "'git+https://github.com/facebookresearch/pytorch3d.git'")
)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def refine(hero_obj_bytes: bytes, mask_png_bytes: bytes, ground_ply_bytes: bytes,
           init: dict, fov_deg: float = 70.0, iters: int = 250, size: int = 512) -> dict:
    import io
    import numpy as np
    import torch
    from PIL import Image
    import trimesh
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer,
        MeshRenderer, SoftSilhouetteShader, BlendParams, look_at_view_transform,
    )

    dev = torch.device("cuda")

    # --- hero mesh, normalised exactly like web/auto.html (centre x/z, feet at y=0) ---
    hv, hf = _load_obj_verts_faces(io.BytesIO(hero_obj_bytes))
    hv = torch.tensor(hv, dtype=torch.float32)
    mn = hv.min(0).values; mx = hv.max(0).values; ctr = (mn + mx) / 2
    hv[:, 0] -= ctr[0]; hv[:, 2] -= ctr[2]; hv[:, 1] -= mn[1]
    base_h = float((mx - mn)[1])                       # PSHuman native height (~3.75)
    hero_v0 = hv.to(dev)
    hero_f = torch.tensor(hf, dtype=torch.int64, device=dev)

    # --- target mask ---
    mask = Image.open(io.BytesIO(mask_png_bytes)).convert("L").resize((size, size))
    tgt = torch.tensor(np.array(mask) > 127, dtype=torch.float32, device=dev)

    # --- ground plane (for contact + a simple collision floor) ---
    g = trimesh.load(io.BytesIO(ground_ply_bytes), file_type="ply", process=False)
    foot0 = np.array([init["x"], init["y"], init["z"]], dtype=np.float32)
    gv = np.asarray(g.vertices)
    near = gv[np.linalg.norm(gv - foot0, axis=1) < 1.5]
    ground_y = float(np.median(near[:, 1])) if len(near) else float(init["y"])

    # --- camera: at origin, looking at the initial foot point, given FOV ---
    # foot0 is float32 numpy; pass plain floats and force the R/T matrices to
    # float32, else PyTorch3D's world->view bmm hits "expected Float, found Double".
    at = tuple(float(v) for v in foot0)
    R, T = look_at_view_transform(eye=((0., 0., 0.),), at=(at,), up=((0., 1., 0.),), device=dev)
    R, T = R.to(torch.float32), T.to(torch.float32)
    cam = FoVPerspectiveCameras(device=dev, R=R, T=T, fov=float(fov_deg))
    blend = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(image_size=size, blur_radius=np.log(1. / 1e-4 - 1.) * blend.sigma,
                                   faces_per_pixel=50)
    renderer = MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=raster),
                            SoftSilhouetteShader(blend_params=blend))

    # --- free parameters: translation, yaw, log-scale (uniform) ---
    # dtype=float32 is load-bearing: np.radians/np.log return numpy float64, and
    # torch.tensor([<np.float64>]) would infer float64, making the transformed
    # verts float64 and tripping PyTorch3D's rasterizer bmm ("Float vs Double").
    t = torch.tensor([init["x"], init["y"], init["z"]], dtype=torch.float32, device=dev, requires_grad=True)
    yaw = torch.tensor([np.radians(init["yaw_deg"])], dtype=torch.float32, device=dev, requires_grad=True)
    target_scale = float(init.get("scale", 1.0))        # auto.html scale = TARGET_H/2 style
    log_s = torch.tensor([np.log(max(target_scale, 1e-3))], dtype=torch.float32, device=dev, requires_grad=True)
    # auto.html applies scale*fit where fit=2/base_h; fold that so world height = s*2.
    fit = 2.0 / base_h

    opt = torch.optim.Adam([t, yaw, log_s], lr=0.02)

    def transform(verts):
        s = torch.exp(log_s) * fit
        c, sn = torch.cos(yaw), torch.sin(yaw)
        Ry = torch.stack([torch.cat([c, torch.zeros_like(c), sn]),
                          torch.tensor([0., 1., 0.], device=dev),
                          torch.cat([-sn, torch.zeros_like(c), c])]).reshape(3, 3)
        return (verts * s) @ Ry.T + t

    for it in range(iters):
        opt.zero_grad()
        vt = transform(hero_v0)
        mesh = Meshes(verts=[vt], faces=[hero_f])
        sil = renderer(mesh)[..., 3]                    # (1,H,W) silhouette alpha
        sil = sil.squeeze(0)
        l_sil = ((sil - tgt) ** 2).mean()
        # ground contact: lowest hero vertex should sit on the ground plane.
        l_contact = (vt[:, 1].min() - ground_y) ** 2
        # non-penetration: penalise hero vertices below the ground plane.
        l_pen = torch.clamp(ground_y - vt[:, 1], min=0).mean()
        # scale prior: stay near the metric target.
        l_scale = (torch.exp(log_s) - target_scale) ** 2
        loss = l_sil + 0.5 * l_contact + 0.5 * l_pen + 0.05 * l_scale
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(f"it {it:3d} loss {loss.item():.4f} sil {l_sil.item():.4f} "
                  f"contact {l_contact.item():.4f}")

    with torch.no_grad():
        out = {
            "x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
            "yaw_deg": float(np.degrees(yaw.item())),
            "scale": float(torch.exp(log_s).item()),
            "ground_y": ground_y, "final_loss": float(loss.item()),
        }
    print("REFINED:", out)
    return out


def _load_obj_verts_faces(fp):
    """Minimal OBJ reader (vertices + triangular faces); ignores colours/UVs."""
    import numpy as np
    V, F = [], []
    for line in io_text(fp):
        if line.startswith("v "):
            p = line.split()
            V.append([float(p[1]), float(p[2]), float(p[3])])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):           # fan-triangulate
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, dtype="float32"), np.asarray(F, dtype="int64")


def io_text(fp):
    for raw in fp.read().decode("utf-8", "ignore").splitlines():
        yield raw


@app.local_entrypoint()
def main(hero: str, mask: str, ground: str, x: float = 0.0, y: float = -0.9,
         z: float = -0.8, yaw_deg: float = 180.0, scale: float = 0.85,
         fov_deg: float = 70.0) -> None:
    import json
    init = {"x": x, "y": y, "z": z, "yaw_deg": yaw_deg, "scale": scale}
    out = refine.remote(open(hero, "rb").read(), open(mask, "rb").read(),
                        open(ground, "rb").read(), init, fov_deg)
    print(json.dumps(out, indent=2))
    print("\nbake into auto.html:\n"
          f"  ?hx={out['x']:.2f}&hy={out['y']:.2f}&hz={out['z']:.2f}"
          f"&hr={out['yaw_deg']:.0f}&hs={out['scale']:.2f}")
