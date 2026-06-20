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


# ── DepthFlow: faithful 2.5D depth-parallax (the "Google Cinematic / Immersity" look)
# for the memory album. Operates on the ORIGINAL photo -> native 1080p+, no
# hallucination. Needs headless OpenGL (EGL) + a depth model (DepthAnythingV2, cached).
df_cache = modal.Volume.from_name("depthflow-cache", create_if_missing=True)
depthflow_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "wget", "ffmpeg", "libglib2.0-0", "fonts-dejavu-core",
        # headless GL: NVIDIA EGL ICD (from the driver) + mesa fallbacks + xvfb safety net
        "libegl1", "libgl1", "libglvnd0", "libgles2", "libegl1-mesa", "libgl1-mesa-dri",
        "libosmesa6", "xvfb", "mesa-utils",
        # depthflow pulls pyaudio (C extension) -> needs a compiler + PortAudio headers
        "build-essential", "portaudio19-dev", "pkg-config",
    )
    .env({
        "HF_HOME": "/dfcache/hf", "TORCH_HOME": "/dfcache/torch",
        "WINDOW_BACKEND": "headless",          # ShaderFlow offscreen backend
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        "MPLBACKEND": "Agg",
        "CC": "gcc", "CXX": "g++",             # add_python reports clang; force gcc
    })
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("depthflow==0.9.0.dev1")
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


def _build_trajectory(pose, video_length):
    """Named camera presets -> a custom-trajectory JSON path. Returns None for raw
    pose strings ('w-31'), which pass through to generate.py unchanged.

    The album defaults use SMALL-amplitude moves that stay near the original
    viewpoint, so the model reveals little new content and barely hallucinates —
    a "living photo" feel rather than a fly-through:
      push     – gentle dolly-in (most faithful: moves *into* the existing scene)
      parallax – gentle lateral truck (depth parallax, minimal reveal)
      gentle   – tiny push + a slight lateral sway that eases back toward origin
      orbit360 – full 360 fly-around (kept, but heavy hallucination; not a default)
    """
    presets = {"orbit360", "push", "parallax", "gentle"}
    if pose not in presets:
        return None
    import json
    import os
    import sys
    import numpy as np
    sys.path.insert(0, WORKDIR)
    from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
    steps = (video_length - 1) // 4          # poses = steps + 1 = latent count
    if pose == "orbit360":
        motions = [{"third_yaw": 2 * np.pi / steps} for _ in range(steps)]
    elif pose == "push":
        motions = [{"forward": 0.012} for _ in range(steps)]
    elif pose == "parallax":
        motions = [{"right": 0.010} for _ in range(steps)]
    else:  # gentle
        motions = [{"forward": 0.006,
                    "right": 0.012 * float(np.cos(np.pi * i / max(steps - 1, 1)))}
                   for i in range(steps)]
    poses = generate_camera_trajectory_local(motions)
    K = [[969.6969696969696, 0.0, 960.0],
         [0.0, 969.6969696969696, 540.0],
         [0.0, 0.0, 1.0]]
    custom = {str(i): {"extrinsic": p.tolist(), "K": K} for i, p in enumerate(poses)}
    os.makedirs(os.path.join(WORKDIR, "assets", "pose"), exist_ok=True)
    path = os.path.join(WORKDIR, "assets", "pose", f"{pose}.json")
    json.dump(custom, open(path, "w"))
    print(f"trajectory '{pose}': {len(poses)} poses -> {path}")
    return path


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
    pose_arg = _build_trajectory(pose, video_length) or pose

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
def stitch(clips: list, captions=None, title=None, fps: int = 24,
           music: bool = True, xfade: float = 0.6) -> str:
    """Stitch the per-photo clips into one album montage with a title card, optional
    per-clip captions, gentle crossfades, and a soft music bed. Probes the clips' real
    dimensions (so the title/captions match) and degrades gracefully: crossfade ->
    hard-cut concat -> raw copy, and music is best-effort (silent if it fails)."""
    import base64
    import json
    import os
    import subprocess
    import tempfile

    def run(args):
        return subprocess.run(args, check=True, capture_output=True)

    def probe(path):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "json", path], capture_output=True, text=True, check=True)
        j = json.loads(r.stdout)
        s = j["streams"][0]
        return int(s["width"]), int(s["height"]), float(j["format"]["duration"])

    d = tempfile.mkdtemp()
    paths = []
    for i, c in enumerate(clips):
        p = os.path.join(d, f"c{i:02d}.mp4")
        open(p, "wb").write(base64.b64decode(c) if isinstance(c, str) else c)
        paths.append(p)

    W, H, _ = probe(paths[0])                 # real clip dims (e.g. 848x480)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    have_font = os.path.exists(font)
    caps = captions or [None] * len(paths)
    out = os.path.join(d, "album.mp4")
    assembled = os.path.join(d, "v.mp4")

    def esc(t):
        return t.replace("\\", "").replace(":", "\\:").replace("'", "")

    # 1) Normalize every segment to identical W,H,fps,SAR,pix_fmt so xfade/concat work.
    norm = "scale=%d:%d,setsar=1,format=yuv420p" % (W, H)
    seq, durs = [], []
    title_dur = 2.0
    if title and have_font:
        tp = os.path.join(d, "title.mp4")
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d={title_dur}:r={fps}",
             "-vf", (f"{norm},drawtext=fontfile={font}:text='{esc(title)}':fontcolor=white:"
                     f"fontsize={int(H*0.09)}:x=(w-tw)/2:y=(h-th)/2:"
                     f"alpha='min(1,min(t/0.5,(%g-t)/0.5))'" % title_dur),
             "-an", tp])
        seq.append(tp); durs.append(title_dur)
    for i, p in enumerate(paths):
        lp = os.path.join(d, f"l{i:02d}.mp4")
        vf = [norm]
        if caps[i] and have_font:
            vf.append(f"drawtext=fontfile={font}:text='{esc(caps[i])}':fontcolor=white:"
                      f"fontsize={int(H*0.055)}:box=1:boxcolor=black@0.4:boxborderw=10:"
                      f"x=(w-tw)/2:y=h-th-{int(H*0.06)}")
        run(["ffmpeg", "-y", "-i", p, "-vf", ",".join(vf), "-r", str(fps), "-an", lp])
        seq.append(lp); durs.append(probe(p)[2])

    # 2) Assemble: crossfade chain -> hard-cut concat fallback.
    try:
        inputs = []
        for s in seq:
            inputs += ["-i", s]
        prev, acc, filt = "[0:v]", durs[0], ""
        for i in range(1, len(seq)):
            off = max(acc - xfade, 0.05)
            filt += f"{prev}[{i}:v]xfade=transition=fade:duration={xfade}:offset={off:.3f}[x{i}];"
            prev = f"[x{i}]"; acc = acc + durs[i] - xfade
        run(["ffmpeg", "-y", *inputs, "-filter_complex", filt.rstrip(";"),
             "-map", prev, "-r", str(fps), "-pix_fmt", "yuv420p", assembled])
    except subprocess.CalledProcessError as e:
        print("xfade failed -> hard-cut concat:", (e.stderr or b"")[-800:].decode("utf-8", "ignore"))
        inputs = []
        for s in seq:
            inputs += ["-i", s]
        fc = "".join(f"[{i}:v]" for i in range(len(seq))) + f"concat=n={len(seq)}:v=1:a=0[o]"
        run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[o]",
             "-r", str(fps), "-pix_fmt", "yuv420p", assembled])

    # 3) Soft warm chord pad (best-effort; silent montage if it fails).
    if music:
        try:
            D = probe(assembled)[2]
            chord = "0.10*sin(2*PI*130.81*t)+0.08*sin(2*PI*164.81*t)+0.06*sin(2*PI*196.0*t)"
            mus = os.path.join(d, "mus.wav")
            run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"aevalsrc={chord}:s=44100:d={D:.2f}",
                 "-af", (f"lowpass=f=2200,afade=t=in:st=0:d=1.2,"
                         f"afade=t=out:st={max(D-1.8,0):.2f}:d=1.8,volume=0.7"), mus])
            run(["ffmpeg", "-y", "-i", assembled, "-i", mus, "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", out])
        except subprocess.CalledProcessError as e:
            print("music mux failed -> silent montage:", (e.stderr or b"")[-800:].decode("utf-8", "ignore"))
            out = assembled
    else:
        out = assembled

    print(f"album: {os.path.getsize(out)} bytes, {W}x{H}, {len(paths)} clips, music={music}")
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


# Gentle, faithful camera moves (small amplitude -> minimal hallucination). We avoid
# the 360 orbit for albums because it forces the model to invent the whole room. We
# alternate a slow push-in and a lateral parallax for variety; both stay near the
# original frame so the photo reads as a "living memory", not a fly-through.
_MOVE_A = "push"        # slow dolly-in
_MOVE_B = "parallax"    # gentle lateral parallax


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
        pose = _MOVE_A if i % 2 == 0 else _MOVE_B  # alternate gentle moves for variety
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


@app.local_entrypoint()
def restitch(clips_dir: str = "demo/album_clips", title: str = "The George on Collins",
             music: bool = True) -> None:
    """CPU-only: re-stitch already-generated clips (title + crossfades + music) without
    re-spending GPU. Iterate the montage/music here cheaply."""
    import base64
    import glob
    import os

    paths = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
    if not paths:
        raise SystemExit(f"no .mp4 clips in {clips_dir}")
    print(f"restitch: {len(paths)} clips from {clips_dir}")
    clips = [base64.b64encode(open(p, "rb").read()).decode() for p in paths]
    album_b64 = stitch.remote(clips, None, title, 24, music)
    os.makedirs("out", exist_ok=True)
    open(os.path.join("out", "album.mp4"), "wb").write(base64.b64decode(album_b64))
    print(f"wrote out/album.mp4 ({os.path.getsize('out/album.mp4')} bytes)")


@app.function(image=depthflow_image, gpu="L4", timeout=1200, scaledown_window=300,
              volumes={"/dfcache": df_cache})
def df_render(image_bytes: bytes, preset: str = "", intensity: float = 0.35,
              width: int = 1920, height: int = 1080, duration: float = 6.0,
              fps: int = 30) -> str:
    """One photo -> a 1080p depth-parallax clip via DepthFlow (headless EGL). Faithful
    'living photo' motion on the real image — no hallucination. Empty preset = the
    tasteful default animation; presets: circle/orbit/dolly/zoom/horizontal/vertical."""
    import base64
    import os
    import subprocess
    import tempfile

    os.environ.setdefault("HF_HOME", "/dfcache/hf")
    df_cache.reload()
    d = tempfile.mkdtemp()
    img = os.path.join(d, "in.png")
    open(img, "wb").write(_prep_image_to_16x9(image_bytes))   # exif + 16:9 fill
    out = os.path.join(d, "out.mp4")

    cmd = ["depthflow", "input", "-i", img]
    if preset:
        cmd += [preset, "--intensity", str(intensity)]
    cmd += ["main", "-o", out, "-w", str(width), "-h", str(height), "-t", str(duration)]
    if fps:
        cmd += ["--fps", str(fps)]
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print((r.stdout or "")[-400:])
    except subprocess.CalledProcessError as e:
        print("EGL render failed -> retry under xvfb. stderr:", (e.stderr or "")[-1500:])
        subprocess.run(["xvfb-run", "-a", *cmd], check=True)
    df_cache.commit()
    print(f"df_render: {os.path.getsize(out)} bytes, {width}x{height}, preset='{preset or 'default'}'")
    return base64.b64encode(open(out, "rb").read()).decode()


@app.local_entrypoint()
def df_test(image: str = "demo/album/03_pair.jpg", preset: str = "") -> None:
    """Validate the DepthFlow/headless-GL path on a single image before a full album."""
    import base64
    import os
    b64 = df_render.remote(open(image, "rb").read(), preset)
    os.makedirs("out", exist_ok=True)
    open("out/df_test.mp4", "wb").write(base64.b64decode(b64))
    print(f"wrote out/df_test.mp4 ({os.path.getsize('out/df_test.mp4')} bytes)")


@app.local_entrypoint()
def run_album_df(images_dir: str = "demo/album", title: str = "The George on Collins") -> None:
    """DepthFlow memory album: each photo -> a 1080p depth-parallax clip (parallel) ->
    one stitched montage with title + crossfades + music. Faithful, no hallucination."""
    import base64
    import glob
    import os

    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                   if p.lower().endswith(exts))
    if not paths:
        raise SystemExit(f"no images found in {images_dir}")
    print(f"depthflow album: {len(paths)} photos from {images_dir}")

    # Alternate the tasteful default move with a gentle circle for subtle variety.
    presets = ["", "circle"]
    jobs = []
    for i, p in enumerate(paths):
        preset = presets[i % len(presets)]
        fc = df_render.spawn(open(p, "rb").read(), preset, 0.35, 1920, 1080, 6.0, 30)
        jobs.append((p, preset, fc))

    os.makedirs("out", exist_ok=True)
    clips = []
    for p, preset, fc in jobs:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            raw = base64.b64decode(fc.get())
            open(os.path.join("out", f"clip_{stem}.mp4"), "wb").write(raw)
            clips.append(raw)
            print(f"  clip OK [{preset or 'default'}] {stem} ({len(raw)} bytes)")
        except Exception as e:
            print(f"  clip FAILED for {p}: {type(e).__name__}: {e}")

    if not clips:
        raise SystemExit("no clips produced — cannot stitch album")
    album = base64.b64decode(stitch.remote(
        [base64.b64encode(c).decode() for c in clips], None, title))
    open(os.path.join("out", "album.mp4"), "wb").write(album)
    print(f"wrote out/album.mp4 from {len(clips)} clips (1080p)")
