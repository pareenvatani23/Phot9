"""
HY-WorldPlay (HY-World 1.5) on Modal — single image -> interactive camera-path
video. A streaming video world model: given one photo (+ optional prompt) and a
camera trajectory, it generates a photorealistic clip moving through that scene,
with the subject native to the world (no mesh, no compositing, no placement).

We run the *distilled autoregressive* model (4 inference steps) on a single
A100-80GB. Per the repo's system-requirements table, the AR-distilled config at
sequence-parallel sp=1 fits in 72 GB — so one 80 GB GPU is enough; multi-GPU only
lowers per-GPU memory / latency.

Build is light vs HunyuanWorld: torch 2.6 / cu124 + the repo's pinned wheels.
SageAttention/FlashAttention are *optional* for the HunyuanVideo backend (the
distilled command runs with --use_sageattn false), so we skip the source builds
and let attention fall back to torch SDPA.

Pipeline: download_models.py (HunyuanVideo-1.5 480P-I2V base + Qwen2.5-VL-7B /
ByT5 / Glyph-SDXL-v2 / FLUX.1-Redux-dev encoders + HY-WorldPlay action ckpts) ->
torchrun hyvideo/generate.py with the distilled AR rollout -> outputs/*.mp4.

NOTE: the FLUX.1-Redux-dev vision encoder is GATED. The HF_TOKEN secret must have
accepted access at https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev,
otherwise the HunyuanVideo pipeline cannot load and we raise a clear error.
"""
from __future__ import annotations

import modal

app = modal.App("phot9-worldplay")

# All HF/ModelScope weights (HunyuanVideo-1.5 base + encoders + HY-WorldPlay
# action models) are cached here once — ~80 GB, downloaded on first run.
weights = modal.Volume.from_name("worldplay-weights", create_if_missing=True)

REPO = "https://github.com/Tencent-Hunyuan/HY-WorldPlay.git"
WORKDIR = "/root/HY-WorldPlay"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", "fonts-dejavu-core")
    .env({
        "HF_HOME": "/weights/hf",          # cache every snapshot on the volume
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # parallel chunked downloads (the first run
                                           # crawled at ~4 MB/s with this off)
        "PYTHONPATH": WORKDIR,
    })
    # torch 2.6 / cu124 first (matches the repo's torch>=2.6 / tv0.21 / ta2.6 pins),
    # so the subsequent -r requirements.txt finds them satisfied and never pulls a
    # CPU build over the CUDA one.
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # The repo + its pinned wheels (diffusers 0.35, transformers 4.56, peft,
    # qwen-vl-utils, modelscope, moviepy, ...). All wheels — no source compiles.
    .run_commands(
        f"git clone --depth 1 {REPO} {WORKDIR}",
        f"pip install -r {WORKDIR}/requirements.txt",
    )
    # hf_transfer is pulled by some hub versions; keep downloads on the plain path.
    .pip_install("hf_transfer")
)


def _have_siglip(hunyuan_path: str) -> bool:
    import os
    d = os.path.join(hunyuan_path, "vision_encoder", "siglip")
    return os.path.isdir(d) and len(os.listdir(d)) > 3


# The downloads are the slow, GPU-irrelevant part (~40-50 GB from HF/ModelScope),
# so they run on a cheap CPU container with a generous timeout. We reuse the repo's
# own encoder-organizing helpers for the HunyuanVideo base + text/vision encoders,
# but fetch ONLY the distilled action model from HY-WorldPlay (the repo's blanket
# snapshot pulls all four ~16 GB variants — bi/ar/ar_rl/ar_distilled — and we need
# just one). Weights are committed to the shared volume for the GPU step to reuse.
@app.function(image=image, timeout=10800, volumes={"/weights": weights})
def fetch_weights(hf_token: str) -> tuple[str, str]:
    import os
    import sys

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    os.environ["HF_HOME"] = "/weights/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    sys.path.insert(0, WORKDIR)
    import download_models as dm
    from huggingface_hub import snapshot_download

    dm.check_dependencies()
    hunyuan_path = dm.download_hunyuan_video()          # vae/scheduler/transformer 480p_i2v
    # Super-resolution uses a *separate* SR transformer + upsampler (720p_sr_distilled)
    # that download_models.py does not fetch. Pull them so --sr true works.
    snapshot_download("tencent/HunyuanVideo-1.5",
                      allow_patterns=["transformer/720p_sr_distilled/*",
                                      "upsampler/720p_sr_distilled/*"])
    dm.download_llm_text_encoder(hunyuan_path)          # Qwen2.5-VL-7B -> text_encoder/llm
    dm.download_byt5_encoders(hunyuan_path)             # byt5-small + Glyph-SDXL-v2
    if hf_token:
        dm.download_vision_encoder(hunyuan_path, hf_token)  # gated FLUX SigLIP

    # Distilled action model only.
    wp = snapshot_download("tencent/HY-WorldPlay",
                           allow_patterns=["ar_distilled_action_model/*"])
    ddir = os.path.join(wp, "ar_distilled_action_model")
    src = os.path.join(ddir, "model.safetensors")
    ar_distill = os.path.join(ddir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(src) and not os.path.exists(ar_distill):
        os.symlink(os.path.realpath(src), ar_distill)   # symlink, don't dup 16 GB

    if not _have_siglip(hunyuan_path):
        raise RuntimeError(
            "Vision encoder (FLUX.1-Redux-dev SigLIP) is missing — the HunyuanVideo "
            "pipeline cannot run without it. Make sure the HF_TOKEN has accepted access "
            "at https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev.")
    if not os.path.exists(ar_distill):
        raise RuntimeError(f"distilled action ckpt not found under {ddir}")

    weights.commit()
    print("FETCH_OK model_path:", hunyuan_path)
    print("FETCH_OK ar_distill:", ar_distill)
    return hunyuan_path, ar_distill


def _prep_image_to_16x9(image_bytes: bytes) -> bytes:
    """Honor EXIF orientation (phone portraits are stored sideways) and normalize to
    16:9 with a blurred-fill background, so the model never crops faces out of frame."""
    import io
    from PIL import Image, ImageOps, ImageFilter
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    W, H = im.size
    target = 16 / 9
    if abs(W / H - target) > 0.02:
        if W / H < target:                       # too tall -> widen
            cw, ch = int(round(H * target)), H
        else:                                    # too wide -> heighten
            cw, ch = W, int(round(W / target))
        bg = ImageOps.fit(im, (cw, ch), method=Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
        fg = ImageOps.contain(im, (cw, ch), method=Image.LANCZOS)
        canvas = bg.copy()
        canvas.paste(fg, ((cw - fg.width) // 2, (ch - fg.height) // 2))
        im = canvas
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _generate_impl(image_bytes, model_path, ar_distill, prompt, pose,
                   video_length, seed, enable_sr) -> dict:
    """Shared body: preprocess -> distilled AR rollout (4 steps) -> mp4(s)."""
    import base64
    import glob
    import os
    import shutil
    import subprocess

    os.environ["HF_HOME"] = "/weights/hf"
    weights.reload()  # ensure this container sees fetch_weights' committed downloads

    if not _have_siglip(model_path):
        raise RuntimeError(f"vision encoder missing under {model_path} — run fetch_weights first")
    if not os.path.exists(ar_distill):
        raise RuntimeError(f"distilled action ckpt not found: {ar_distill}")

    inp = os.path.join(WORKDIR, "input.png")
    open(inp, "wb").write(_prep_image_to_16x9(image_bytes))   # exif + 16:9 fill
    out_dir = os.path.join(WORKDIR, "outputs")
    shutil.rmtree(out_dir, ignore_errors=True)   # warm containers reuse this dir across
    os.makedirs(out_dir, exist_ok=True)          # clips — clear so our glob is this run's

    # Resolve the camera path. `pose` is either a pose string ("w-31"), an existing
    # .json path, or the "orbit360" preset -> a closed-loop 360° orbit (third_yaw)
    # built with the repo's own trajectory generator. A full 360 returns the camera
    # exactly to the start pose, so the last frame reframes the opening photo.
    pose_arg = pose
    if pose.startswith("orbit"):
        import json
        import numpy as np
        import sys as _sys
        _sys.path.insert(0, WORKDIR)
        from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
        latents = (video_length - 1) // 4 + 1           # poses needed (incl. frame 0)
        steps = latents - 1
        motions = [{"third_yaw": 2 * np.pi / steps} for _ in range(steps)]
        poses = generate_camera_trajectory_local(motions)
        K = [[969.6969696969696, 0.0, 960.0],
             [0.0, 969.6969696969696, 540.0],
             [0.0, 0.0, 1.0]]                            # repo's reference intrinsic
        custom = {str(i): {"extrinsic": p.tolist(), "K": K} for i, p in enumerate(poses)}
        os.makedirs(os.path.join(WORKDIR, "assets", "pose"), exist_ok=True)
        pose_arg = os.path.join(WORKDIR, "assets", "pose", "orbit360.json")
        json.dump(custom, open(pose_arg, "w"))
        print(f"orbit360: {len(poses)} poses, {np.degrees(2*np.pi/steps):.1f}°/latent -> {pose_arg}")

    cmd = [
        "torchrun", "--nproc_per_node=1", "--master_port=29517",
        "hyvideo/generate.py",
        "--prompt", prompt,
        "--image_path", inp,
        "--resolution", "480p",
        "--aspect_ratio", "16:9",
        "--video_length", str(video_length),
        "--seed", str(seed),
        "--rewrite", "false",
        "--sr", ("true" if enable_sr else "false"), "--save_pre_sr_video",
        "--pose", pose_arg,
        "--output_path", out_dir,
        "--model_path", model_path,
        "--action_ckpt", ar_distill,
        "--few_step", "true",
        "--num_inference_steps", "4",
        "--model_type", "ar",
        "--use_vae_parallel", "false",
        "--use_sageattn", "false",
        "--use_fp8_gemm", "false",
        "--transformer_resident_ar_rollout", "true",
    ]
    env = {**os.environ, "PYTHONPATH": WORKDIR}
    subprocess.run(cmd, cwd=WORKDIR, check=True, env=env)

    mp4s = sorted(glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True))
    print("WORLDPLAY_OUTPUTS:", mp4s)
    if not mp4s:
        raise RuntimeError("no .mp4 produced — check generate.py logs above")
    # Prefer the super-resolved clip (gen_sr.mp4) as the primary deliverable; fall
    # back to the largest file if SR was off or named differently.
    sr = [p for p in mp4s if os.path.basename(p) == "gen_sr.mp4"]
    primary = sr[0] if sr else max(mp4s, key=lambda p: os.path.getsize(p))
    ordered = [primary] + [p for p in mp4s if p != primary]
    result = {}
    for i, p in enumerate(ordered):
        name = "worldplay.mp4" if i == 0 else os.path.basename(p)
        result[name] = base64.b64encode(open(p, "rb").read()).decode()
        print(f"  {name}: {os.path.getsize(p)} bytes  ({p})")
    return result


@app.function(image=image, gpu="H200", timeout=3600, scaledown_window=600,
              volumes={"/weights": weights})
def generate(image_bytes: bytes, model_path: str, ar_distill: str,
             prompt: str = "Cinematic camera moving smoothly through the scene, photorealistic, natural lighting.",
             pose: str = "w-31", video_length: int = 125, seed: int = 1,
             enable_sr: bool = True) -> dict:
    # HD/hero clips: H200 has headroom for base + 720p SR (which OOMs an 80 GB GPU).
    return _generate_impl(image_bytes, model_path, ar_distill, prompt, pose,
                          video_length, seed, enable_sr)


@app.function(image=image, gpu="A100-80GB", timeout=3600, scaledown_window=600,
              volumes={"/weights": weights})
def generate_draft(image_bytes: bytes, model_path: str, ar_distill: str,
                   prompt: str = "Cinematic camera moving smoothly through the scene, photorealistic, natural lighting.",
                   pose: str = "w-31", video_length: int = 125, seed: int = 1) -> dict:
    # Album drafts: 480p, no SR, on a cheaper/more-available A100 so N photos fan out
    # in parallel. HD is the per-album upgrade (generate() on H200).
    return _generate_impl(image_bytes, model_path, ar_distill, prompt, pose,
                          video_length, seed, False)


@app.function(image=image, timeout=1200)
def stitch(clips: list, captions=None, title=None, fps: int = 24) -> str:
    """Concatenate the per-photo clips into one album montage (ffmpeg). Burns an
    optional title card + per-clip captions; falls back to a plain hard-cut concat if
    the captioned path fails for any reason, so an album always renders."""
    import base64
    import os
    import subprocess
    import tempfile

    d = tempfile.mkdtemp()
    paths = []
    for i, c in enumerate(clips):
        p = os.path.join(d, f"c{i:02d}.mp4")
        open(p, "wb").write(base64.b64decode(c) if isinstance(c, str) else c)
        paths.append(p)
    W, H = 832, 480
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    have_font = os.path.exists(font)
    out = os.path.join(d, "album.mp4")
    caps = captions or [None] * len(paths)

    def esc(t):
        return t.replace("\\", "").replace(":", "\\:").replace("'", "")

    try:
        seq = []
        if title and have_font:
            tp = os.path.join(d, "title.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d=1.8:r={fps}",
                 "-vf", (f"drawtext=fontfile={font}:text='{esc(title)}':fontcolor=white:"
                         f"fontsize=44:x=(w-tw)/2:y=(h-th)/2,fade=t=in:st=0:d=0.4,"
                         f"fade=t=out:st=1.4:d=0.4"),
                 "-pix_fmt", "yuv420p", "-an", tp], check=True, capture_output=True)
            seq.append(tp)
        for i, p in enumerate(paths):
            lp = os.path.join(d, f"l{i:02d}.mp4")
            vf = []
            if caps[i] and have_font:
                vf.append(f"drawtext=fontfile={font}:text='{esc(caps[i])}':fontcolor=white:"
                          f"fontsize=26:box=1:boxcolor=black@0.45:boxborderw=10:"
                          f"x=(w-tw)/2:y=h-th-28")
            vf.append("fade=t=in:st=0:d=0.3")
            subprocess.run(["ffmpeg", "-y", "-i", p, "-vf", ",".join(vf), "-r", str(fps),
                            "-pix_fmt", "yuv420p", "-an", lp], check=True, capture_output=True)
            seq.append(lp)
        inputs = []
        for s in seq:
            inputs += ["-i", s]
        n = len(seq)
        fc = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
        subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[out]",
                        "-r", str(fps), "-pix_fmt", "yuv420p", out], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("captioned stitch failed -> plain concat. stderr:",
              (e.stderr or b"")[-1500:].decode("utf-8", "ignore"))
        listf = os.path.join(d, "list.txt")
        open(listf, "w").write("".join(f"file '{p}'\n" for p in paths))
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf,
                        "-c", "copy", out], check=True)
    print(f"album: {os.path.getsize(out)} bytes from {len(paths)} clips")
    return base64.b64encode(open(out, "rb").read()).decode()


@app.local_entrypoint()
def run_demo(image: str = "demo/hiker.jpg",
             prompt: str = "Cinematic camera moving smoothly through the scene, photorealistic, natural lighting.",
             pose: str = "w-31", video_length: int = 125, seed: int = 1) -> None:
    import base64
    import os

    hf = os.environ.get("HF_TOKEN", "")
    # Slow ~40-50 GB download on a cheap CPU container (idempotent, cached on the
    # volume); then the GPU step just loads + infers.
    model_path, ar_distill = fetch_weights.remote(hf)
    data = open(image, "rb").read()
    # Graceful degradation: requested length+SR -> 125+SR (SR on a long clip can OOM
    # an 80 GB GPU) -> 125 no-SR (always fits). Keeps super-resolution whenever memory
    # allows, and still guarantees a clip.
    plan, seen = [], set()
    for vl, sr in [(video_length, True), (125, True), (125, False)]:
        if (vl, sr) not in seen:
            seen.add((vl, sr)); plan.append((vl, sr))
    result = None
    for idx, (vl, sr) in enumerate(plan):
        try:
            result = generate.remote(data, model_path, ar_distill, prompt, pose, vl, seed, sr)
            print(f"generate OK at video_length={vl}, sr={sr}")
            break
        except Exception as e:
            print(f"attempt video_length={vl}, sr={sr} failed ({type(e).__name__}: {e})")
            if idx == len(plan) - 1:
                raise
    os.makedirs("out", exist_ok=True)
    for name, b64 in result.items():
        path = os.path.join("out", name)
        open(path, "wb").write(base64.b64decode(b64))
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")


# Auto camera move per photo (heuristic v1; VLM-driven in the product): orbit wide
# table/scene shots to show the space, dolly into portraits/close subjects.
_PORTRAIT_MOVE = "w-31"      # gentle dolly-in
_SCENE_MOVE = "orbit360"     # closed-loop fly-around


@app.local_entrypoint()
def run_album(images_dir: str = "demo/album", title: str = "Memory Album",
              seed: int = 1) -> None:
    """Fan-out orchestrator: every photo in images_dir -> a clip (in parallel) ->
    one stitched album montage. Writes out/clip_*.mp4 and out/album.mp4."""
    import base64
    import glob
    import os

    hf = os.environ.get("HF_TOKEN", "")
    model_path, ar_distill = fetch_weights.remote(hf)

    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                   if p.lower().endswith(exts))
    if not paths:
        raise SystemExit(f"no images found in {images_dir}")
    print(f"album: {len(paths)} photos from {images_dir}")

    prompt = ("Cinematic camera moving smoothly through the scene, photorealistic, "
              "warm natural restaurant lighting.")

    # Fan out: spawn one draft (A100, 480p) job per photo; they run in parallel.
    jobs = []
    for i, p in enumerate(paths):
        pose = _SCENE_MOVE if i % 2 == 0 else _PORTRAIT_MOVE  # alternate for variety
        data = open(p, "rb").read()
        fc = generate_draft.spawn(data, model_path, ar_distill, prompt, pose, 125, seed)
        jobs.append((p, pose, fc))

    os.makedirs("out", exist_ok=True)
    clips = []
    for p, pose, fc in jobs:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            res = fc.get()
            raw = base64.b64decode(res["worldplay.mp4"])
            cp = os.path.join("out", f"clip_{stem}.mp4")
            open(cp, "wb").write(raw)
            clips.append(raw)
            print(f"  clip OK [{pose}] {cp} ({len(raw)} bytes)")
        except Exception as e:
            print(f"  clip FAILED for {p}: {type(e).__name__}: {e}")

    if not clips:
        raise SystemExit("no clips produced — cannot stitch album")

    album_b64 = stitch.remote([base64.b64encode(c).decode() for c in clips], None, title)
    open(os.path.join("out", "album.mp4"), "wb").write(base64.b64decode(album_b64))
    print(f"wrote out/album.mp4 from {len(clips)} clips")
