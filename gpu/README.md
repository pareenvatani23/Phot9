# Phot9 GPU pipeline (Modal)

The GPU half of the "full volumetric" diorama path: one photo → walkable 3D
Gaussian-splat scene with a **sharp** person. Marble (hosted) builds the world;
these GPU steps build the person and composite them in with correct scale and
ground contact.

Modal is used so we can deploy/run GPU functions from a CPU-only CI runner —
execution happens on Modal's GPUs, billed per-second with scale-to-zero.

## What you provision (one-time)

1. **Modal account** → `modal token new` → set repo secrets
   `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. (Free monthly credits to start;
   ~$2.50/A100-hr after.)
2. **Marble paid plan** → `WORLDLABS_API_KEY` (already used by `backend/`; needed
   for `.ply` world export).
3. Nothing else — no GPU box to manage.

## Phases (each is a kill-gate)

| Phase | Function          | Status | What it proves / does |
|-------|-------------------|--------|-----------------------|
| 1     | `smoke`           | ✅ green | GPU (L40S), CUDA + matmul, egress HF/WorldLabs 200 |
| 2     | `human_pass`      | stub   | photo → person as Gaussians (HumanSplat → SAM-3D-Body → PSHuman) |
| 3     | `solve_placement` | stub   | depth + Marble camera → human transform |
| 4     | `composite`       | stub   | merge + contact-refine → `.spz` for Spark |

## Run Phase 1

Locally:

```bash
pip install -r gpu/requirements.txt
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal run gpu/app.py::smoke
```

Or dispatch the **GPU pipeline (Phase 1 smoke)** GitHub Action with the Modal
token — it mirrors the existing `marble.yml` dispatch pattern.

Expect output like `PHOT9_GPU_SMOKE {... 'cuda_available': True, 'device':
'NVIDIA L40S', 'matmul_ok': True, 'egress': {...}}`.

## Licensing watch-out

Phase-2/3 models (HumanSplat, SAM-3D-Body, PSHuman) are research-licensed. If
this becomes commercial, swap to MIT/Apache parts (NoPoSplat, gsplat). Decision
deferred per the build plan; default for now is best quality.
