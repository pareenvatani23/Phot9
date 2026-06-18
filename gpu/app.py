"""
Phot9 GPU pipeline (Modal).

This is the GPU half of the "full volumetric" diorama path. Modal lets us author
and deploy GPU functions from a CPU-only CI runner while execution happens on
Modal's GPUs — so the existing GitHub Actions / fal flow can drive it without us
managing a box.

Pipeline (one photo -> walkable 3DGS scene with a sharp person):

    photo
     |-- Marble World API ............ .ply scene world + camera   (hosted, backend/)
     |-- human_pass()    [GPU] ....... person  -> Gaussian splats  (Phase 2)
     |-- solve_placement() [GPU] ..... depth + camera -> transform (Phase 3)
     |-- composite()     [GPU] ....... merge + contact-refine      (Phase 4)
     `-- -> .spz -> Spark viewer ...... web/index.html             (have)

Phase 1 (this file, runnable now): `smoke` proves we can get a GPU, that CUDA is
live, and that the container has outbound network — before any model work.

Run (after `modal token set ...`):
    modal run gpu/app.py::smoke

Phases 2-4 are explicit stubs below; each is gated so we can kill the project at
any checkpoint without sunk cost.
"""

from __future__ import annotations

import modal

app = modal.App("phot9-gpu")

# Phase 1 image is deliberately light (fast cold build). Heavy model deps
# (human-splat checkpoints, pytorch3d, etc.) are added per-phase so the smoke
# test stays cheap to iterate on.
smoke_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "requests==2.32.3")
)


@app.function(image=smoke_image, gpu="L40S", timeout=300)
def smoke() -> dict:
    """Phase 1 infra smoke test: GPU present, CUDA live, egress works."""
    import torch
    import requests

    info: dict = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    }

    # A trivial GPU op so we know the device actually executes, not just enumerates.
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        info["matmul_ok"] = bool(torch.isfinite((x @ x).sum()).item())

    # Egress check — the GPU container must reach Marble / fal / HF to pull
    # checkpoints and post results in later phases.
    egress = {}
    for name, url in {
        "huggingface": "https://huggingface.co",
        "worldlabs": "https://platform.worldlabs.ai",
        "fal": "https://fal.run",
    }.items():
        try:
            egress[name] = requests.head(url, timeout=10).status_code
        except Exception as exc:  # noqa: BLE001 - report, don't fail the probe
            egress[name] = f"ERR {type(exc).__name__}"
    info["egress"] = egress

    print("PHOT9_GPU_SMOKE", info)
    return info


# --------------------------------------------------------------------------- #
# Phase 2 — Human splat pass (GATED: implement after Phase 1 is green)
# --------------------------------------------------------------------------- #
@app.function(gpu="A100", timeout=1200)
def human_pass(image_url: str) -> str:
    """photo -> person as 3D Gaussians (.ply on a Modal volume; return URL/key).

    Model ladder (best quality first; fall back if it won't containerize):
      1. HumanSplat  — single image -> human Gaussians, native splat compositing.
      2. SAM-3D-Body — Meta MHR mesh recovery (robust; needs splat conversion).
      3. PSHuman     — multiview human gen with identity/face-distortion fixes.

    Licensing: all three are research-licensed. Swap to MIT/Apache parts only if
    this goes commercial (decision deferred per build plan).
    """
    raise NotImplementedError("Phase 2 — gated on Phase 1 smoke test passing")


# --------------------------------------------------------------------------- #
# Phase 3 — Placement solver (GATED)
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", timeout=600)
def solve_placement(human_ply: str, scene_ply: str, image_url: str) -> dict:
    """Solve the human's similarity transform into Marble's camera frame.

    Monocular humans are scale-ambiguous; the fix is to fit scale+offset to the
    scene's KNOWN camera (Marble provides it) using a monocular depth estimate
    (Depth-Anything) at the person's ground-contact point. Returns a 4x4 (+scale).
    """
    raise NotImplementedError("Phase 3 — gated on Phase 2")


# --------------------------------------------------------------------------- #
# Phase 4 — Composite + contact refine (GATED)
# --------------------------------------------------------------------------- #
@app.function(gpu="A100", timeout=900)
def composite(human_ply: str, scene_ply: str, transform: dict) -> str:
    """Merge human splats into the scene and contact-refine (AHA-style) so feet
    neither float nor sink, then export .spz for the Spark viewer."""
    raise NotImplementedError("Phase 4 — gated on Phase 3")
