"""
Orbit video -> photoreal 3D Gaussian Splat (.splat) on Modal, with PPISP.

Multi-view pipeline (no license-gated assets — open code + your own video):
    1. video -> frames (ffmpeg)
    2. COLMAP SfM -> camera poses + sparse cloud
    3. rembg subject masks (so the busy backdrop doesn't fuse into the body)
    4. NVIDIA 3DGUT training with PPISP post-processing
       (post_processing.method=ppisp) -> compensates exposure / white-balance /
       vignetting drift across the orbit frames
    5. export scene.ply -> scene.splat (antimatter15 packing, what web/ Spark
       viewer + the ci-media convention already consume) + a turntable mp4

Two entrypoints, matching the cheap-then-expensive pattern:
    modal run gpu/splat_app.py::preview   --video demo/orbit.mp4   # pose sanity check
    modal run gpu/splat_app.py::run_demo  --video demo/orbit.mp4   # full train

Outputs are returned base64 (same contract as pshuman_app.py) and written to
./out by the local entrypoint; CI force-pushes ./out to the ci-splat branch and
the Spark viewer loads it via web/world.html?world=<raw-url>/scene.splat .
"""

from __future__ import annotations

import modal

app = modal.App("phot9-splat")

# Persist frames / COLMAP / training runs across invocations so the FULL stage
# reuses what PREVIEW already solved instead of redoing SfM.
work = modal.Volume.from_name("splat-work", create_if_missing=True)

# 3DGUT installs into its own uv venv; we must invoke THAT interpreter for
# training (the base python only carries our preprocessing deps).
VENV = "/opt/3dgrut/.venv/bin/python"
VPIP = "/opt/3dgrut/.venv/bin/pip"

image = (
    # CUDA 12.8 devel (3DGRUT default) so nvcc can build its CUDA extensions on
    # cheap CPU builders rather than a billed GPU.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "wget", "ffmpeg", "colmap",
        "libgl1-mesa-dev", "libglib2.0-0", "libegl1", "libgles2",
        "build-essential", "cmake", "ninja-build",
    )
    .env({
        "QT_QPA_PLATFORM": "offscreen",      # headless COLMAP
        "PYOPENGL_PLATFORM": "egl",          # headless orbit render
        "MPLBACKEND": "Agg",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9",
        "FORCE_CUDA": "1",
    })
    .run_commands(
        "git clone --recursive https://github.com/nv-tlabs/3dgrut.git /opt/3dgrut",
        # Build the 3DGUT env (uv venv + compiled CUDA ext). The heavy first-build
        # step; iterate here if wheels/versions drift.
        "cd /opt/3dgrut && pip install uv && ./install_env_uv.sh",
        # 3DGUT JIT-compiles its CUDA kernels via Slang at first run. Without
        # slangtorch (which bundles the slangc compiler) it looks for a standalone
        # `slangc` on PATH and dies with FileNotFoundError. Install it in the venv.
        f"{VPIP} install slangtorch",
        # PPISP package into 3DGUT's venv (post_processing.method=ppisp needs it).
        f"{VPIP} install ppisp || {VPIP} install git+https://github.com/nv-tlabs/ppisp.git",
    )
    # Preprocessing + export deps into the BASE python (not the venv).
    .pip_install(
        "rembg[gpu]", "onnxruntime-gpu", "numpy", "plyfile",
        "imageio", "imageio-ffmpeg", "matplotlib", "pillow",
    )
)


# ---------------------------------------------------------------------------
# Shared, idempotent stages (reuse on-disk so FULL doesn't redo PREVIEW).
# ---------------------------------------------------------------------------
def _paths(name):
    import pathlib
    root = pathlib.Path("/work") / name
    return root, root / "images", root / "sparse"


def _frames(root, images, video_bytes, fps):
    import subprocess
    have = list(images.glob("frame_*.jpg"))
    if have:
        print(f"[frames] reuse {len(have)}"); return len(have)
    images.mkdir(parents=True, exist_ok=True)
    (root / "src.mp4").write_bytes(video_bytes)
    subprocess.run(["ffmpeg", "-y", "-i", str(root / "src.mp4"),
                    "-vf", f"fps={fps}", "-qscale:v", "2",
                    str(images / "frame_%04d.jpg")], check=True, capture_output=True)
    n = len(list(images.glob("frame_*.jpg")))
    print(f"[frames] extracted {n}")
    if n < 20:
        raise RuntimeError(f"only {n} frames — orbit too short / fps too low")
    return n


def _colmap(root, images, sparse):
    import subprocess
    if (sparse / "0").exists():
        print("[colmap] reuse sparse"); return
    sparse.mkdir(parents=True, exist_ok=True)
    db = root / "colmap.db"
    run = lambda a: subprocess.run(a, check=True, capture_output=True, text=True)
    # CPU SIFT (use_gpu 0): COLMAP's GPU SiftGPU needs an OpenGL/EGL context,
    # which a headless Modal GPU container lacks -> SIGABRT. CPU is headless-safe
    # and fast enough at this frame count.
    run(["colmap", "feature_extractor", "--database_path", str(db),
         "--image_path", str(images), "--ImageReader.single_camera", "1",
         "--SiftExtraction.use_gpu", "0"])
    run(["colmap", "exhaustive_matcher", "--database_path", str(db),
         "--SiftMatching.use_gpu", "0"])
    run(["colmap", "mapper", "--database_path", str(db),
         "--image_path", str(images), "--output_path", str(sparse)])
    if not (sparse / "0").exists():
        raise RuntimeError("COLMAP registered no cameras — capture too fast / "
                           "blurry / low-overlap. See CAPTURE guidance.")


def _ply_to_splat(ply_path, splat_path):
    """3DGS .ply -> antimatter15 .splat (32 bytes/gaussian: pos f32*3,
    scale f32*3, color u8*4, quat u8*4). Sorted by opacity*size desc."""
    import numpy as np
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    rot /= (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)
    SH_C0 = 0.28209479177387814
    color = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    rgba = np.clip(np.concatenate(
        [color, opacity[:, None]], 1) * 255, 0, 255).astype(np.uint8)
    order = np.argsort(-(opacity * scale.prod(1)))
    buf = bytearray()
    rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
    for i in order:
        buf += xyz[i].tobytes() + scale[i].tobytes() + rgba[i].tobytes() + rot_u8[i].tobytes()
    open(splat_path, "wb").write(buf)
    return len(order)


def _b64(path):
    import base64
    return base64.b64encode(open(path, "rb").read()).decode()


# ---------------------------------------------------------------------------
# PREVIEW — cheap, stops after the camera solve. Returns montage + trajectory.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", volumes={"/work": work}, timeout=60 * 20)
def preview_infer(video_bytes: bytes, name: str = "subject", fps: float = 4.0) -> dict:
    import subprocess, re, math, pathlib
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    root, images, sparse = _paths(name)
    n = _frames(root, images, video_bytes, fps)
    _colmap(root, images, sparse)

    # stats via colmap model_analyzer
    a = subprocess.run(["colmap", "model_analyzer", "--path", str(sparse / "0")],
                       capture_output=True, text=True)
    txt = a.stdout + a.stderr
    grab = lambda lbl, c=float: (c(re.search(rf"{lbl}:\s*([0-9.]+)", txt).group(1))
                                 if re.search(rf"{lbl}:\s*([0-9.]+)", txt) else None)
    reg = grab("Registered images", int)
    reproj = grab("Mean reprojection error")

    # camera centers (model -> TXT)
    td = root / "sparse_txt"; td.mkdir(exist_ok=True)
    subprocess.run(["colmap", "model_converter", "--input_path", str(sparse / "0"),
                    "--output_path", str(td), "--output_type", "TXT"],
                   check=True, capture_output=True, text=True)
    lines = [l for l in (td / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    C = []
    for i in range(0, len(lines), 2):
        e = lines[i].split(); qw, qx, qy, qz = map(float, e[1:5])
        t = np.array(list(map(float, e[5:8])))
        R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                      [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                      [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
        C.append(-R.T @ t)
    C = np.array(C) - np.mean(C, 0)

    out = root / "preview"; out.mkdir(exist_ok=True)
    # trajectory plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for axx, (i, j, lab) in zip(ax, [(0, 2, "top-down X-Z"), (0, 1, "side X-Y")]):
        axx.scatter(C[:, i], C[:, j], c=np.arange(len(C)), cmap="viridis", s=18)
        axx.plot(C[:, i], C[:, j], lw=.5, alpha=.4); axx.set_aspect("equal")
        axx.set_title(lab); axx.grid(alpha=.3)
    fig.suptitle(f"{reg}/{n} cams · reproj {reproj}px")
    fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=110); plt.close(fig)
    # montage
    frames = sorted(images.glob("frame_*.jpg")); pick = frames[::max(1, len(frames)//6)][:6]
    sheet = Image.new("RGB", (3*320, 2*240), (20, 20, 20))
    for k, f in enumerate(pick):
        sheet.paste(Image.open(f).convert("RGB").resize((320, 240)),
                    ((k % 3)*320, (k//3)*240))
    sheet.save(out / "montage.jpg")
    work.commit()

    verdict = ("GOOD — proceed to full" if reg and reg >= 0.8*n and (reproj or 9) < 1.5
               else "MARGINAL — inspect trajectory/montage before training")
    return {"frames": n, "registered": reg, "reproj_px": reproj, "verdict": verdict,
            "trajectory.png": _b64(out / "trajectory.png"),
            "montage.jpg": _b64(out / "montage.jpg")}


# ---------------------------------------------------------------------------
# FULL — reuse frames+poses, mask, train 3DGUT+PPISP, export splat + orbit.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", volumes={"/work": work}, timeout=60 * 60)
def reconstruct(video_bytes: bytes, name: str = "subject", fps: float = 4.0,
                use_ppisp: bool = True, mask: bool = False,
                n_iters: int = 7000) -> dict:
    import subprocess, pathlib
    root, images, sparse = _paths(name)

    # Idempotent: if this name was already trained+exported (persisted on the
    # volume), just republish the splat instead of paying to retrain.
    cached = root / "out" / "scene.splat"
    if cached.exists():
        print("[reuse] scene.splat already on volume — skipping training")
        res = {"scene.splat": _b64(cached), "reused": True}
        cp = root / "out" / "orbit.mp4"
        if cp.exists():
            res["orbit.mp4"] = _b64(cp)
        return res

    _frames(root, images, video_bytes, fps)
    _colmap(root, images, sparse)

    # Optional human-subject masks (added AFTER colmap so SfM ignores them).
    # Off by default: only enable for a single-person subject — for a general
    # scene the human segmenter would wrongly blank everything out.
    if mask:
        from rembg import remove, new_session
        from PIL import Image
        sess = new_session("u2net_human_seg")
        for f in sorted(images.glob("frame_*.jpg")):
            m = images / (f.stem + "_mask.png")
            if not m.exists():
                remove(Image.open(f).convert("RGBA"), session=sess, only_mask=True).save(m)

    run_dir = root / "runs"
    cmd = [VENV, "train.py", "--config-name", "apps/colmap_3dgut.yaml",
           f"path={root}", f"out_dir={run_dir}", f"experiment_name={name}",
           "dataset.downsample_factor=1",
           # Cap iterations: 3DGUT defaults to n_iterations=30000 (the 3DGS
           # standard), which is ~30+ min on an A10G. 7000 is its first
           # checkpoint -- a solid splat at ~4x less GPU time/cost.
           f"n_iterations={n_iters}",
           # 3DGUT saves .pt checkpoints, not .ply, unless this is on. Enable so
           # we get the gaussian-splat .ply to convert to scene.splat.
           "export_ply.enabled=true"]
    if use_ppisp:
        cmd.append("post_processing.method=ppisp")
    p = subprocess.run(cmd, cwd="/opt/3dgrut", capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        print(p.stderr[-4000:]); raise RuntimeError("3DGUT training failed; see logs")

    out = root / "out"; out.mkdir(exist_ok=True)
    plys = sorted(run_dir.rglob("*.ply"), key=lambda q: q.stat().st_mtime)
    if not plys:
        raise RuntimeError("no .ply produced by training")
    import shutil
    shutil.copy(plys[-1], out / "scene.ply")
    n_g = _ply_to_splat(out / "scene.ply", out / "scene.splat")
    work.commit()
    res = {"gaussians": n_g, "ppisp": use_ppisp,
           "scene.splat": _b64(out / "scene.splat")}
    mp4s = sorted(run_dir.rglob("*orbit*.mp4")) or sorted(run_dir.rglob("*.mp4"))
    if mp4s:
        shutil.copy(mp4s[-1], out / "orbit.mp4"); res["orbit.mp4"] = _b64(out / "orbit.mp4")
    return res


# ---------------------------------------------------------------------------
# Local entrypoints (CI calls these; results written to ./out).
# ---------------------------------------------------------------------------
def _dump(result, outdir="out"):
    import os, base64
    os.makedirs(outdir, exist_ok=True)
    for k, val in result.items():
        if isinstance(val, str) and ("." in k):       # base64 file payloads
            open(os.path.join(outdir, k), "wb").write(base64.b64decode(val))
            print(f"wrote {outdir}/{k}")
        else:
            print(f"{k}: {val}")


@app.local_entrypoint()
def preview(video: str = "demo/orbit.mp4", name: str = "summit", fps: float = 4.0):
    """Cheap camera-solve sanity check before paying for training."""
    res = preview_infer.remote(open(video, "rb").read(), name, fps)
    _dump(res, "out")


@app.local_entrypoint()
def run_demo(video: str = "demo/orbit.mp4", name: str = "summit",
             fps: float = 4.0, ppisp: bool = True, mask: bool = False,
             iters: int = 7000):
    """Full reconstruction; reuses preview's frames+poses if present."""
    res = reconstruct.remote(open(video, "rb").read(), name, fps, ppisp, mask, iters)
    _dump(res, "out")
