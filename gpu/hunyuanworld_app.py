"""
HunyuanWorld-1.0 on Modal — single image -> explorable 3D world (layered textured
meshes, mesh_layer*.ply). Self-hosted alternative to World Labs Marble: no
per-world cost, no credit limits, and mesh output means no Gaussian-splat spikes.

Heavy build (CUDA 12.4 / torch 2.5; pytorch3d + flash-attn + MoGe + Real-ESRGAN +
ZIM). Lessons from the PSHuman build are applied up front: force g++ (Modal's
add_python reports clang), single CUDA arch (A100=8.0), --no-build-isolation for
torch-dependent source builds, and a prebuilt flash-attn wheel to skip its compile.

Pipeline: demo_panogen.py (image->panorama) then demo_scenegen.py (pano->world).
"""
from __future__ import annotations

import modal

app = modal.App("phot9-hunyuanworld")

# Weights cache (gated tencent/HunyuanWorld-1 + ZIM onnx) — download once.
weights = modal.Volume.from_name("hunyuanworld-weights", create_if_missing=True)

REPO = "https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git"
WORKDIR = "/root/HunyuanWorld-1.0"
# Prebuilt flash-attn wheel for torch2.5 / cu12 / cp310 (skips the long compile).
FLASH_WHL = ("https://github.com/Dao-AILab/flash-attention/releases/download/"
             "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "cmake", "build-essential", "ninja-build", "wget",
        "libgl1", "libglib2.0-0", "libegl1", "libgles2", "ffmpeg",
    )
    # g++ (not the standalone-python clang), single A100 arch, build-from-source CUDA.
    .env({
        "CC": "gcc", "CXX": "g++",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "HF_HOME": "/weights/hf",
    })
    .pip_install(
        "torch==2.5.0", "torchvision==0.20.0", "torchaudio==2.5.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("setuptools==69.5.1", "wheel", "ninja", "numpy==1.24.1")
    # flash-attn via prebuilt wheel (no compile)
    .run_commands(f"pip install {FLASH_WHL}")
    # pytorch3d from source (the long compile) — isolate so it caches on its own.
    .run_commands(
        "pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'",
    )
    # The rest of HunyuanWorld's pinned pip deps (mostly wheels — fast).
    .pip_install(
        "transformers==4.51.0", "diffusers==0.34.0", "accelerate",
        "onnxruntime-gpu==1.21.1", "pytorch-lightning==2.4.0",
        "xformers==0.0.28.post2", "timm==1.0.13", "kornia==0.8.0",
        "ultralytics==8.3.74", "opencv-python==4.11.0.86", "trimesh",
        "open3d", "einops", "omegaconf", "huggingface_hub",
    )
    .pip_install("git+https://github.com/microsoft/MoGe.git")
    # The repo + ZIM + Real-ESRGAN.
    .run_commands(
        f"git clone {REPO} {WORKDIR}",
        f"git clone https://github.com/naver-ai/ZIM.git {WORKDIR}/ZIM && "
        f"cd {WORKDIR}/ZIM && pip install -e .",
        f"git clone https://github.com/xinntao/Real-ESRGAN.git {WORKDIR}/Real-ESRGAN && "
        f"cd {WORKDIR}/Real-ESRGAN && pip install basicsr-fixed facexlib gfpgan && "
        f"pip install -r requirements.txt && python setup.py develop",
    )
    # ZIM ONNX weights (public, no token) baked in.
    .run_commands(
        f"mkdir -p {WORKDIR}/zim_vit_l_2092 && cd {WORKDIR}/zim_vit_l_2092 && "
        "wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx && "
        "wget -q https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx",
    )
    # transformers 4.51 requires huggingface-hub <1.0; an unpinned install pulled
    # 1.20. Pin it last so the final env is compatible (cheap appended layer).
    .pip_install("huggingface_hub==0.34.0")
    # basicsr imports torchvision.transforms.functional_tensor, removed in tv>=0.17.
    # Re-create it as a shim re-exporting from functional (the classic basicsr fix).
    .run_commands(
        "TVDIR=$(python -c 'import torchvision, os; print(os.path.dirname(torchvision.__file__))') && "
        "echo 'from torchvision.transforms.functional import rgb_to_grayscale' > \"$TVDIR/transforms/functional_tensor.py\" && "
        "echo SHIM_OK"
    )
    # Small utility deps the conda env pulls that aren't in the core wheels above.
    # peft is required for the FLUX LoRA adapters HunyuanWorld loads.
    .pip_install(
        "easydict", "scipy", "scikit-image", "imageio", "imageio-ffmpeg",
        "matplotlib", "plyfile", "py360convert", "sentencepiece",
        "open_clip_torch", "ftfy", "rembg", "pymeshlab", "peft", "protobuf",
    )
    # HunyuanWorld uses the pre-rewrite flat utils3d API (utils3d.numpy.image_uv and
    # the utils3d.np alias with create_icosahedron_mesh). MoGe's transitive pin (1.3)
    # removed all of it and pulls numpy 2.x. Install the flat-API commit (--no-deps to
    # protect numpy 1.24), add the np alias it lacks, and verify both funcs at build.
    .run_commands(
        "pip install --force-reinstall --no-deps "
        "git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f && "
        "python -c \"import utils3d, os; "
        "open(os.path.join(os.path.dirname(utils3d.__file__), '__init__.py'), 'a').write('\\nfrom utils3d import numpy as np\\n')\" && "
        "python -c \"import utils3d; "
        "print('IMG', hasattr(utils3d.numpy, 'image_uv'), 'ICO', hasattr(utils3d.numpy, 'create_icosahedron_mesh')); "
        "print('NPALIAS_OK', utils3d.np.create_icosahedron_mesh)\""
    )
)


@app.function(image=image, gpu="A100-80GB", timeout=3600,
              volumes={"/weights": weights})
def generate(image_bytes: bytes, hf_token: str,
             classes: str = "outdoor", fg1: str = "stones", fg2: str = "trees") -> dict:
    import base64
    import glob
    import os
    import subprocess

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_TOKEN"] = hf_token

    # Gated HunyuanWorld weights -> cached volume, symlinked where the repo expects.
    ckpt = "/weights/HunyuanWorld-1"
    if not glob.glob(ckpt + "/*"):
        from huggingface_hub import snapshot_download
        # Raises a clear GatedRepoError if the HF license hasn't been accepted.
        snapshot_download("tencent/HunyuanWorld-1", local_dir=ckpt, token=hf_token or None)
        weights.commit()
    link = os.path.join(WORKDIR, "HunyuanWorld-1")
    if not os.path.islink(link):
        subprocess.run(["ln", "-sfn", ckpt, link], check=True)
    # scenegen looks for ZIM weights at ./ZIM/zim_vit_l_2092; they're baked at the
    # repo root — symlink them into place.
    zim_link = os.path.join(WORKDIR, "ZIM", "zim_vit_l_2092")
    if not os.path.exists(zim_link):
        subprocess.run(["ln", "-sfn", os.path.join(WORKDIR, "zim_vit_l_2092"), zim_link], check=True)

    os.makedirs(os.path.join(WORKDIR, "examples", "in"), exist_ok=True)
    inp = os.path.join(WORKDIR, "examples", "in", "input.png")
    open(inp, "wb").write(image_bytes)
    out = os.path.join(WORKDIR, "test_results", "in")

    # Stage 1: image -> panorama
    subprocess.run(
        ["python3", "demo_panogen.py", "--prompt", "", "--image_path", inp, "--output_path", out],
        cwd=WORKDIR, check=True,
    )
    # Stage 2: panorama -> layered 3D world (meshes)
    subprocess.run(
        ["python3", "demo_scenegen.py",
         "--image_path", os.path.join(out, "panorama.png"),
         "--labels_fg1", fg1, "--labels_fg2", fg2, "--classes", classes,
         "--output_path", out],
        cwd=WORKDIR, check=True,
    )

    plys = sorted(glob.glob(os.path.join(out, "**", "*.ply"), recursive=True))
    print("HUNYUAN_OUTPUTS:", plys)
    return {os.path.basename(p): base64.b64encode(open(p, "rb").read()).decode() for p in plys}


@app.local_entrypoint()
def run_demo(image: str = "demo/hiker.jpg", classes: str = "outdoor",
             fg1: str = "stones", fg2: str = "trees") -> None:
    import base64
    import os

    hf = os.environ.get("HF_TOKEN", "")
    data = open(image, "rb").read()
    result = generate.remote(data, hf, classes, fg1, fg2)
    os.makedirs("out", exist_ok=True)
    if not result:
        print("WARNING: no .ply meshes returned")
    for name, b64 in result.items():
        path = os.path.join("out", name)
        open(path, "wb").write(base64.b64decode(b64))
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
