"""Extract a raycast-friendly ground mesh from a HunyuanWorld world.

HunyuanWorld wraps the scene in a large sky sphere; rays cast to find the ground
hit that dome unless it's removed. This drops the sky shell (faces whose centroid
is near the max vertex radius) and keeps the foreground/background terrain, so the
auto-placement solver in web/auto.html can raycast real ground.

Usable two ways:
  - standalone:  python gpu/extract_ground.py <in.ply> [more.ply ...] -o world_ground.ply
  - imported:    from extract_ground import build_ground_mesh  (used by hunyuanworld_app)

trimesh is required; open3d (optional) is only used for decimation when present.
"""
from __future__ import annotations

SKY_FRAC = 0.8        # keep faces whose centroid radius < SKY_FRAC * max-vertex-radius
DECIMATE_TARGET = 800_000


def build_ground_mesh(ply_paths, out_path, sky_frac=SKY_FRAC, target=DECIMATE_TARGET):
    """Merge the given meshes, drop the sky sphere, decimate, write out_path.

    Returns (n_vertices, n_faces) of the written mesh, or None if nothing remained.
    """
    import numpy as np
    import trimesh

    meshes = []
    for p in ply_paths:
        m = trimesh.load(p, process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        if len(getattr(m, "faces", [])):
            meshes.append(m)
    if not meshes:
        return None
    merged = trimesh.util.concatenate(meshes)

    V = merged.vertices
    sky_r = float(np.linalg.norm(V, axis=1).max())
    fc = V[merged.faces].mean(axis=1)
    keep = np.linalg.norm(fc, axis=1) < sky_r * sky_frac
    if not keep.any():
        return None
    ground = trimesh.Trimesh(vertices=V, faces=merged.faces[keep],
                             vertex_colors=getattr(merged.visual, "vertex_colors", None),
                             process=True)
    ground.remove_unreferenced_vertices()

    # Optional decimation (open3d) so the published mesh stays web-friendly.
    if len(ground.faces) > target:
        try:
            import open3d as o3d
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(ground.vertices),
                o3d.utility.Vector3iVector(ground.faces))
            if ground.visual.vertex_colors is not None:
                cols = (ground.visual.vertex_colors[:, :3] / 255.0)
                m.vertex_colors = o3d.utility.Vector3dVector(cols)
            m = m.simplify_quadric_decimation(target)
            ground = trimesh.Trimesh(
                vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles),
                vertex_colors=(np.asarray(m.vertex_colors) * 255).astype("uint8")
                if len(m.vertex_colors) else None, process=False)
        except Exception as e:  # decimation is best-effort
            print("ground decimation skipped:", repr(e))

    ground.export(out_path)
    print(f"world_ground: sky_r={sky_r:.2f} kept {len(ground.faces)} faces "
          f"({100*keep.mean():.0f}% of merged), wrote {out_path}")
    return len(ground.vertices), len(ground.faces)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="world .ply file(s) (merged or per-layer)")
    ap.add_argument("-o", "--out", default="world_ground.ply")
    ap.add_argument("--sky-frac", type=float, default=SKY_FRAC)
    ap.add_argument("--target", type=int, default=DECIMATE_TARGET)
    a = ap.parse_args()
    r = build_ground_mesh(a.inputs, a.out, a.sky_frac, a.target)
    print("done" if r else "no ground produced")
