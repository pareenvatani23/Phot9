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
    .apt_install("git", "wget", "ffmpeg", "libgl1", "libglib2.0-0")
    .env({
        "HF_HOME": "/weights/hf",          # cache every snapshot on the volume
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
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


@app.function(image=image, gpu="A100-80GB", timeout=5400,
              volumes={"/weights": weights})
def generate(image_bytes: bytes, hf_token: str,
             prompt: str = "Cinematic camera moving smoothly through the scene, photorealistic, natural lighting.",
             pose: str = "w-31", video_length: int = 125, seed: int = 1) -> dict:
    import base64
    import glob
    import os
    import subprocess

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    os.environ["HF_HOME"] = "/weights/hf"

    # 1) Fetch/organize every required model (idempotent — skips what's cached).
    #    Needs FLUX access on the token for the gated vision encoder.
    subprocess.run(
        ["python3", "download_models.py", "--hf_token", hf_token],
        cwd=WORKDIR, check=True,
    )
    weights.commit()

    # 2) Resolve the paths run.sh expects (download_models.py prints these).
    from huggingface_hub import snapshot_download
    model_path = snapshot_download("tencent/HunyuanVideo-1.5", local_files_only=True)
    worldplay_path = snapshot_download("tencent/HY-WorldPlay", local_files_only=True)
    ar_distill = os.path.join(
        worldplay_path, "ar_distilled_action_model", "diffusion_pytorch_model.safetensors")

    if not _have_siglip(model_path):
        raise RuntimeError(
            "Vision encoder (FLUX.1-Redux-dev SigLIP) is missing — the HunyuanVideo "
            "pipeline cannot run without it. Make sure the HF_TOKEN has accepted access "
            "at https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev.")
    if not os.path.exists(ar_distill):
        raise RuntimeError(f"distilled action ckpt not found: {ar_distill}")

    # 3) Write the input image and run the distilled AR rollout (4 steps, sp=1).
    inp = os.path.join(WORKDIR, "input.png")
    open(inp, "wb").write(image_bytes)
    out_dir = os.path.join(WORKDIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

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
        "--sr", "false", "--save_pre_sr_video",
        "--pose", pose,
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
    # Return the largest clip first (the full / post-SR video) under a stable name.
    mp4s.sort(key=lambda p: os.path.getsize(p), reverse=True)
    result = {}
    for i, p in enumerate(mp4s):
        name = "worldplay.mp4" if i == 0 else os.path.basename(p)
        result[name] = base64.b64encode(open(p, "rb").read()).decode()
        print(f"  {name}: {os.path.getsize(p)} bytes  ({p})")
    return result


@app.local_entrypoint()
def run_demo(image: str = "demo/hiker.jpg",
             prompt: str = "Cinematic camera moving smoothly through the scene, photorealistic, natural lighting.",
             pose: str = "w-31", video_length: int = 125, seed: int = 1) -> None:
    import base64
    import os

    hf = os.environ.get("HF_TOKEN", "")
    data = open(image, "rb").read()
    result = generate.remote(data, hf, prompt, pose, video_length, seed)
    os.makedirs("out", exist_ok=True)
    for name, b64 in result.items():
        path = os.path.join("out", name)
        open(path, "wb").write(base64.b64decode(b64))
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
