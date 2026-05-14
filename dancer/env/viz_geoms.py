"""Visualise Asimov (and optionally G1) collision + visual geoms in Rerun.

Loads each robot's description into a Newton model, runs forward kinematics
on the rest pose, then logs every collision primitive/mesh and every visual
STL to a Rerun viewer. Used for debugging contact behaviour — e.g. why
Newton's contact solver explodes on Asimov's thin foot capsules and not on
G1's volume-mesh feet.

Usage:
    .venv/bin/python -m dancer.env.viz_geoms                 # Asimov only
    .venv/bin/python -m dancer.env.viz_geoms --with-g1       # both, side by side
    .venv/bin/python -m dancer.env.viz_geoms --save out.rrd  # write to file

Identification of visual vs collision is via `shape_flags` (works for both
MJCF and URDF loaders):
    bit 1 (VISIBLE)            → visual
    bit 2 (COLLIDE_SHAPES) or
    bit 4 (COLLIDE_GROUND)     → collision
"""

from __future__ import annotations

import argparse
from pathlib import Path

import newton
import numpy as np
import rerun as rr
import warp as wp


ASIMOV_XML = Path("data/robot/asimov-v1/xmls/asimov.xml")
G1_URDF = Path("data/robot/g1/g1_29dof_rev_1_0.urdf")

# Newton GeoType values (verified against newton.GeoType).
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 1, 3, 4, 5, 6, 7, 8

# Newton ShapeFlags bits we care about.
FLAG_VISIBLE = 1
FLAG_COLLIDE_SHAPES = 2
FLAG_COLLIDE_GROUND = 4

# Per-robot offsets so they don't overlap when shown together.
# G1's URDF is fixed-base (no freejoint), so its root link sits at the URDF
# origin (z=0) and its feet would clip ~78 cm below the floor. Lift it to
# its natural standing height.
ASIMOV_OFFSET = np.array([0.0, -0.7, 0.0])
G1_OFFSET     = np.array([0.0, +0.7, 0.785])


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw)
# ---------------------------------------------------------------------------

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=a.dtype)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    t = 2.0 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


def _compose(body_pose: np.ndarray, shape_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compose body world pose with shape body-local pose. Both 7-vec (pos, xyzw)."""
    b_pos, b_q = body_pose[:3], body_pose[3:7]
    s_pos, s_q = shape_local[:3], shape_local[3:7]
    return b_pos + _quat_rotate(b_q, s_pos), _quat_mul(b_q, s_q)


def _short(label: str, n: int = 2) -> str:
    parts = label.split("/")
    return "/".join(parts[-n:]) if len(parts) >= n else label


# ---------------------------------------------------------------------------
# Per-robot loading + logging
# ---------------------------------------------------------------------------

def _load_robot(name: str, builder_fn) -> tuple[newton.Model, newton.State]:
    builder = newton.ModelBuilder()
    builder_fn(builder)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def _log_robot(model: newton.Model, state: newton.State,
               *, name: str, offset: np.ndarray,
               coll_color, vis_color) -> tuple[int, int]:
    """Log all collision + visual geoms for one robot under `world/<name>/`.

    `coll_color`, `vis_color` are RGBA tuples.
    Returns (n_collision, n_visual).
    """
    body_q = state.body_q.numpy()        # Warp array → numpy directly,
                                          # avoiding wp.to_torch (which imports
                                          # torch and trips a NCCL mismatch).
    shape_body = model.shape_body.numpy()
    shape_T = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_type = model.shape_type.numpy()
    shape_flags = model.shape_flags.numpy() if hasattr(model.shape_flags, "numpy") else np.array(model.shape_flags)
    shape_labels = list(model.shape_label)
    shape_source = model.shape_source

    cap_centers, cap_qs, cap_lens, cap_radii = [], [], [], []
    box_centers, box_qs, box_halfsizes = [], [], []
    sphere_centers, sphere_radii = [], []
    coll_meshes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    vis_meshes:  list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for i in range(model.shape_count):
        bi = int(shape_body[i])
        if bi < 0:
            continue  # ground plane / world

        flags = int(shape_flags[i])
        is_visual = bool(flags & FLAG_VISIBLE)
        is_collision = bool(flags & (FLAG_COLLIDE_SHAPES | FLAG_COLLIDE_GROUND))
        if not is_visual and not is_collision:
            continue

        w_pos, w_q = _compose(body_q[bi], shape_T[i])
        w_pos = w_pos + offset                                      # apply per-robot lateral offset
        t = int(shape_type[i])
        sc = shape_scale[i]
        lbl = shape_labels[i]

        if is_collision:
            if t == CAPSULE:
                # rr.Capsules3D puts its origin at the FIRST end-cap centre
                # and extends `length` along the entity's local +Z, NOT at
                # the midpoint. Newton/MJCF stores the midpoint as
                # shape_transform.pos, so we shift by -half_length along the
                # capsule's world-frame axis (= world_q @ (0,0,1)).
                length = float(2.0 * sc[1])
                world_z = _quat_rotate(w_q, np.array([0.0, 0.0, 1.0]))
                from_endpoint = w_pos - 0.5 * length * world_z
                cap_centers.append(from_endpoint.astype(np.float32))
                cap_qs.append(w_q.astype(np.float32))
                cap_lens.append(length)
                cap_radii.append(float(sc[0]))
            elif t == BOX:
                box_centers.append(w_pos.astype(np.float32))
                box_qs.append(w_q.astype(np.float32))
                box_halfsizes.append(sc.astype(np.float32))
            elif t == SPHERE:
                sphere_centers.append(w_pos.astype(np.float32))
                sphere_radii.append(float(sc[0]))
            elif t == CYLINDER:
                # Same first-endpoint origin shift as CAPSULE above.
                length = float(2.0 * sc[1])
                world_z = _quat_rotate(w_q, np.array([0.0, 0.0, 1.0]))
                from_endpoint = w_pos - 0.5 * length * world_z
                cap_centers.append(from_endpoint.astype(np.float32))
                cap_qs.append(w_q.astype(np.float32))
                cap_lens.append(length)
                cap_radii.append(float(sc[0]))
            elif t == MESH:
                src = shape_source[i]
                if src is None or getattr(src, "vertices", None) is None:
                    continue
                verts = np.asarray(src.vertices, dtype=np.float32)
                inds = np.asarray(src.indices, dtype=np.uint32).reshape(-1, 3)
                coll_meshes.append((lbl, w_pos.astype(np.float32),
                                    w_q.astype(np.float32), verts, inds))
        elif is_visual:
            if t == MESH:
                src = shape_source[i]
                if src is None or getattr(src, "vertices", None) is None:
                    continue
                verts = np.asarray(src.vertices, dtype=np.float32)
                inds = np.asarray(src.indices, dtype=np.uint32).reshape(-1, 3)
                vis_meshes.append((lbl, w_pos.astype(np.float32),
                                   w_q.astype(np.float32), verts, inds))

    # ---- Log primitives ----
    if cap_centers:
        rr.log(
            f"world/{name}/collision/capsules",
            rr.Capsules3D(
                lengths=cap_lens, radii=cap_radii,
                colors=[coll_color] * len(cap_centers),
            ),
            rr.InstancePoses3D(
                translations=cap_centers,
                quaternions=[rr.Quaternion(xyzw=q) for q in cap_qs],
            ),
            static=True,
        )
    if box_centers:
        rr.log(
            f"world/{name}/collision/boxes",
            rr.Boxes3D(half_sizes=box_halfsizes,
                       colors=[coll_color] * len(box_centers)),
            rr.InstancePoses3D(
                translations=box_centers,
                quaternions=[rr.Quaternion(xyzw=q) for q in box_qs],
            ),
            static=True,
        )
    if sphere_centers:
        rr.log(
            f"world/{name}/collision/spheres",
            rr.Points3D(positions=sphere_centers, radii=sphere_radii,
                        colors=[coll_color] * len(sphere_centers)),
            static=True,
        )
    for lbl, w_pos, w_q, verts, inds in coll_meshes:
        path = f"world/{name}/collision/mesh_{_short(lbl)}"
        rr.log(path, rr.Transform3D(translation=w_pos,
                                    rotation=rr.Quaternion(xyzw=w_q)),
               static=True)
        rr.log(path, rr.Mesh3D(vertex_positions=verts,
                               triangle_indices=inds,
                               albedo_factor=coll_color),
               static=True)

    for lbl, w_pos, w_q, verts, inds in vis_meshes:
        path = f"world/{name}/visual/{_short(lbl)}"
        rr.log(path, rr.Transform3D(translation=w_pos,
                                    rotation=rr.Quaternion(xyzw=w_q)),
               static=True)
        rr.log(path, rr.Mesh3D(vertex_positions=verts,
                               triangle_indices=inds,
                               albedo_factor=vis_color),
               static=True)

    n_coll = (len(cap_centers) + len(box_centers)
              + len(sphere_centers) + len(coll_meshes))
    return n_coll, len(vis_meshes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-g1", action="store_true",
                   help="Also load and render G1 alongside Asimov.")
    p.add_argument("--no-asimov", action="store_true",
                   help="Skip the Asimov robot (useful with --with-g1 to see G1 only).")
    p.add_argument("--no-visual", action="store_true",
                   help="Skip logging visual STLs (only collisions).")
    p.add_argument("--save", type=Path, default=None,
                   help="Write to .rrd file instead of spawning viewer.")
    p.add_argument("--app-id", type=str, default="dancer-viz-geoms")
    args = p.parse_args()

    wp.set_device("cpu")

    if args.save is not None:
        rr.init(args.app_id)
        rr.save(str(args.save))
    else:
        rr.init(args.app_id, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/floor",
        rr.Boxes3D(centers=[[0.0, 0.0, -0.005]],
                   half_sizes=[[1.5, 1.5, 0.005]],
                   colors=[[60, 70, 80, 100]]),
        static=True,
    )

    # Color palette per robot. Visuals are kept very low-alpha so the
    # collision primitives remain visible through the mesh.
    asimov_coll = [255, 80, 80, 220]      # red, mostly opaque
    asimov_vis  = [180, 180, 180, 50]     # grey, ~20% alpha
    g1_coll     = [80, 255, 140, 220]     # green
    g1_vis      = [200, 200, 230, 50]     # pale blue-grey, ~20% alpha

    if not args.no_asimov:
        model, state = _load_robot(
            "asimov",
            lambda b: b.add_mjcf(str(ASIMOV_XML), floating=None,
                                 enable_self_collisions=False),
        )
        # Only apply the side-by-side lateral offset when G1 is also rendered;
        # solo Asimov stays at world origin so positions match MuJoCo's viewer.
        offset = ASIMOV_OFFSET if args.with_g1 else np.zeros(3)
        n_c, n_v = _log_robot(
            model, state, name="asimov", offset=offset,
            coll_color=asimov_coll,
            vis_color=asimov_vis if not args.no_visual else None,
        )
        print(f"asimov: {n_c} collision shapes, {n_v} visual meshes")

    if args.with_g1:
        model, state = _load_robot(
            "g1",
            lambda b: b.add_urdf(str(G1_URDF)),
        )
        n_c, n_v = _log_robot(
            model, state, name="g1", offset=G1_OFFSET,
            coll_color=g1_coll,
            vis_color=g1_vis if not args.no_visual else None,
        )
        print(f"g1: {n_c} collision shapes, {n_v} visual meshes")

    if args.save is not None:
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
