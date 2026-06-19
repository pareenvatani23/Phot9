"""
Clean-plate generator: remove the person(s) from a photo and inpaint the hole,
so the world generator (Marble / HunyuanWorld) never bakes a blurry person into
the scene. Runs on CPU (GitHub runner) — no GPU needed.

  rembg (u2net_human_seg) -> person mask -> dilate -> LaMa inpaint -> clean plate

Usage: python gpu/cleanplate.py --image demo/hiker.jpg --out out/clean.png
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
from simple_lama_inpainting import SimpleLama


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-out", default="out/person_mask.png",
                    help="tight person mask (white=person) for placement refinement")
    ap.add_argument("--dilate", type=int, default=21, help="grow the mask to cover edges/shadow")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")

    # Person mask (white = person). Human-seg model catches all people in frame.
    mask = remove(img, only_mask=True, session=new_session("u2net_human_seg"))
    tight = (np.array(mask) > 30).astype(np.uint8) * 255
    # Save the tight (un-dilated) mask: gpu/refine_placement.py matches the hero
    # silhouette against it. The dilated copy below is only for inpainting coverage.
    if args.mask_out:
        os.makedirs(os.path.dirname(args.mask_out) or ".", exist_ok=True)
        Image.fromarray(tight).save(args.mask_out)
        print(f"PERSON_MASK_OK wrote {args.mask_out} "
              f"({100 * (tight > 0).mean():.1f}% of pixels)")
    m = tight
    if args.dilate > 0:
        m = cv2.dilate(m, np.ones((args.dilate, args.dilate), np.uint8), iterations=1)
    print(f"mask coverage: {100 * (m > 0).mean():.1f}% of pixels")

    # LaMa fills the masked region from surrounding context.
    clean = SimpleLama()(img, Image.fromarray(m))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    clean.convert("RGB").save(args.out)
    print(f"CLEAN_PLATE_OK wrote {args.out} {clean.size}")


if __name__ == "__main__":
    main()
