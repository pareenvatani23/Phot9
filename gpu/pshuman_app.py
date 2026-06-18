"""
PSHuman on Modal — single image -> textured 3D human mesh (.obj), on A100-80GB.

Built from the confirmed recipe (official pengHTYX/PSHuman repo + the working
fffiloni Space pins). The public Space is broken, so we run it ourselves where we
control the environment and get real tracebacks.

Heavy, fragile build (compiles nvdiffrast + pytorch3d + kaolin against torch
2.1.0/cu121) and needs >40 GB VRAM -> A100-80GB. Checkpoints are cached in a
Modal Volume so weights download only once.

Run (after Modal credits + token):
    modal run gpu/pshuman_app.py::run_demo --image demo/crops/boat_woman.png

Output: vertex-colored .obj files in ./out (final = result_clr_scale*_*.obj).
"""

from __future__ import annotations

import modal

app = modal.App("phot9-pshuman")

# Persistent cache so the multi-GB HF checkpoints (PSHuman unclip + SD2.1-unclip
# + rembg u2net) download only on the first run.
hf_cache = modal.Volume.from_name("pshuman-hf-cache", create_if_missing=True)

REPO = "https://github.com/pengHTYX/PSHuman.git"
KAOLIN_FIND_LINKS = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html"

image = (
    # CUDA 12.1 *devel* base so nvcc is present to compile nvdiffrast/pytorch3d.
    # PSHuman's env is Python 3.10.
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "build-essential", "ninja-build",
        "libgl1", "libglib2.0-0", "libegl1", "libgles2", "libgomp1",
        "libosmesa6-dev", "freeglut3-dev", "ffmpeg",
    )
    # torch FIRST (repo README order), pinned to the cu121 build the rest target.
    .pip_install(
        "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # kaolin from NVIDIA's wheel index for exactly torch-2.1.0_cu121.
    .run_commands(f"pip install kaolin==0.17.0 -f {KAOLIN_FIND_LINKS}")
    # Clone the repo, then install its requirements (this is the slow part:
    # nvdiffrast + pytorch3d compile from the git pins inside requirements.txt).
    .run_commands(
        f"git clone {REPO} /root/PSHuman",
        "cd /root/PSHuman && pip install -r requirements.txt",
        gpu="A100",  # some ops probe CUDA at build time; give the builder a GPU
    )
    # Non-gated SMPL-X / ECON assets mirror (avoids the smpl-x.is.tue.mpg.de gate).
    # Lands under /root/PSHuman/data to match the expected data/ layout.
    .run_commands(
        "huggingface-cli download fffiloni/PSHuman-SMPL-related "
        "--repo-type model --local-dir /root/PSHuman/data"
    )
    .env({"PYOPENGL_PLATFORM": "egl", "HF_HOME": "/root/.cache/huggingface"})
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def pshuman_infer(image_bytes: bytes, fname: str = "input.png",
                  crop_size: int = 740, seed: int = 600) -> dict:
    """Run PSHuman on one image; return {filename: base64(obj)} for every .obj."""
    import base64
    import glob
    import os
    import subprocess

    workdir = "/root/PSHuman"
    examples = os.path.join(workdir, "examples")
    os.makedirs(examples, exist_ok=True)
    # Start clean so we only collect this run's outputs.
    for d in ("out", "examples"):
        subprocess.run(["rm", "-rf", os.path.join(workdir, d)], check=False)
    os.makedirs(examples, exist_ok=True)
    with open(os.path.join(examples, fname), "wb") as f:
        f.write(image_bytes)

    # Stage 1: background removal -> white-bg RGBA the model expects.
    subprocess.run(
        ["python", "utils/remove_bg.py", "--path", "examples"],
        cwd=workdir, check=True,
    )

    # Stage 2: cross-scale multiview diffusion + explicit remeshing.
    subprocess.run(
        [
            "python", "inference.py",
            "--config", "configs/inference-768-6view.yaml",
            "pretrained_model_name_or_path=pengHTYX/PSHuman_Unclip_768_6views",
            f"validation_dataset.crop_size={crop_size}",
            "with_smpl=false",
            "validation_dataset.root_dir=examples",
            f"seed={seed}",
            "num_views=7",
            "save_mode=rgb",
        ],
        cwd=workdir, check=True,
    )
    hf_cache.commit()

    objs = sorted(glob.glob(os.path.join(workdir, "out", "**", "*.obj"), recursive=True))
    print(f"PSHUMAN_OUTPUTS: {objs}")
    return {
        os.path.basename(p): base64.b64encode(open(p, "rb").read()).decode()
        for p in objs
    }


@app.local_entrypoint()
def run_demo(image: str = "demo/crops/boat_woman.png",
             crop_size: int = 740, seed: int = 600) -> None:
    """Read a repo image, run remotely, write returned .obj files to ./out."""
    import base64
    import os

    data = open(image, "rb").read()
    result = pshuman_infer.remote(data, os.path.basename(image), crop_size, seed)
    os.makedirs("out", exist_ok=True)
    if not result:
        print("WARNING: no .obj files returned")
    for name, b64 in result.items():
        path = os.path.join("out", name)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
