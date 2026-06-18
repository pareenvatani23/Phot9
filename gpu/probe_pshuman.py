"""
Free PSHuman quality probe via the live HuggingFace Space `fffiloni/PSHuman`.

The Space runs PSHuman on HF's own (ZeroGPU) hardware, so this costs us no Modal
GPU time — it's a quality check before we commit to the heavy A100 build.

Two modes:
  --mode introspect : print the Space's gradio API (endpoint names + arg list),
                      WITHOUT triggering a GPU run. Cheap; tells us the exact
                      api_name + parameters so the run step isn't a guess.
  --mode run        : push one image through and copy back every returned file
                      (mesh .obj/.glb + turntable .mp4) into --out.

ZeroGPU Spaces strongly prefer an HF token (set HF_TOKEN); anonymous calls may be
queued or refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from gradio_client import Client, handle_file


def _collect_files(result, out_dir: str) -> list[str]:
    """Recursively find existing file paths in a gradio result and copy them."""
    found: list[str] = []

    def walk(x):
        if isinstance(x, str) and os.path.exists(x) and os.path.isfile(x):
            found.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(result)
    saved = []
    for p in found:
        dst = os.path.join(out_dir, os.path.basename(p))
        shutil.copy(p, dst)
        saved.append(dst)
        print(f"saved -> {dst} ({os.path.getsize(dst)} bytes)")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="fffiloni/PSHuman")
    ap.add_argument("--mode", choices=["introspect", "run"], required=True)
    ap.add_argument("--image")
    ap.add_argument("--out", default="out")
    ap.add_argument("--api-name", default=None, help="exact api_name from introspect")
    ap.add_argument("--args-json", default="[]", help="extra positional args after the image, as JSON list")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    print(f"Connecting to {args.space} (hf_token={'set' if token else 'none'}) ...")
    client = Client(args.space, hf_token=token)

    if args.mode == "introspect":
        # Human-readable signature (this is what we actually need to read).
        print("=== view_api ===")
        client.view_api(all_endpoints=True)
        return

    if not args.image:
        raise SystemExit("--image is required in run mode")
    os.makedirs(args.out, exist_ok=True)

    extra = json.loads(args.args_json)
    kwargs = {"api_name": args.api_name} if args.api_name else {}
    print(f"predict(image={args.image}, extra={extra}, {kwargs}) ...")
    result = client.predict(handle_file(args.image), *extra, **kwargs)
    print("RAW RESULT:", result)

    saved = _collect_files(result, args.out)
    if not saved:
        print("WARNING: no file outputs found in the result")


if __name__ == "__main__":
    main()
