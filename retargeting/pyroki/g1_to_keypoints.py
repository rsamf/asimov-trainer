"""Convert a LAFAN1 G1 CSV into a pyroki-compatible keypoint .npy.

The LAFAN1 G1 CSVs already encode root + 29 joint angles in G1's coordinate
system. Running G1 forward kinematics on each frame gives us body world poses,
from which we can extract 15 SMPL-style keypoints + 3 auxiliary points in the
exact schema the pyroki retargeter expects.

This is stage 1 of the LAFAN1 → Asimov retargeting pipeline; the output
.npy is then consumed by `retarget_to_asimov.py`.

Output schema (positions are SMPL-style keypoints we extract via G1 FK):
    positions:           (T, 18, 3) float32
    orientations:        (T, 18, 3, 3) float32   row-major rotation matrices
    left_foot_contacts:  (T, 2) int    [ankle, foot]
    right_foot_contacts: (T, 2) int    [ankle, foot]

The 15 base keypoints, in order:
    pelvis, left_hip, right_hip, left_knee, right_knee,
    left_ankle, right_ankle, left_foot, right_foot,
    left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist.

Plus 3 aux: left_hand_aux, right_hand_aux, torso_aux (the same auxiliary
points the pyroki cost terms expect at indices 15-17).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from ..g1 import load_g1_model
from ..joint_map import G1_CSV_JOINT_ORDER


# Conceptual keypoint name -> G1 URDF body name.
# G1 has no separate toe/foot link, so left_foot/right_foot are derived from
# the ankle_roll_link plus a forward offset (handled below).
G1_KEYPOINT_LINKS: list[tuple[str, str | None]] = [
    # MuJoCo strips G1's URDF root link (`pelvis`) when loading, fusing it
    # with the world. So the pelvis keypoint comes directly from the CSV's
    # root pose, not from FK — represented as None here.
    ("pelvis",         None),
    ("left_hip",       "left_hip_pitch_link"),
    ("right_hip",      "right_hip_pitch_link"),
    ("left_knee",      "left_knee_link"),
    ("right_knee",     "right_knee_link"),
    ("left_ankle",     "left_ankle_roll_link"),
    ("right_ankle",    "right_ankle_roll_link"),
    ("left_foot",      None),  # ankle + forward offset
    ("right_foot",     None),
    ("left_shoulder",  "left_shoulder_pitch_link"),
    ("right_shoulder", "right_shoulder_pitch_link"),
    ("left_elbow",     "left_elbow_link"),
    ("right_elbow",    "right_elbow_link"),
    ("left_wrist",     "left_wrist_yaw_link"),
    ("right_wrist",    "right_wrist_yaw_link"),
]
N_RETARGET = 15

# Local-frame offsets (in the parent link's frame) for derived keypoints.
FOOT_LOCAL_OFFSET = np.array([0.15, 0.0, -0.03])  # forward + slightly below ankle
LEFT_HAND_AUX_OFFSET = np.array([0.0, 0.0, 0.20])  # tip of forearm in wrist frame
RIGHT_HAND_AUX_OFFSET = np.array([0.0, 0.0, 0.20])
TORSO_AUX_LOCAL = np.array([0.15, 0.0, -0.10])     # forward+down from torso link
TORSO_LINK_NAME = "torso_link"

# Foot-contact heuristic thresholds.
CONTACT_HEIGHT = 0.10   # m above ground
CONTACT_VEL    = 0.30   # m/s


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def _xmat_to_rotmat(xmat_flat9: np.ndarray) -> np.ndarray:
    """MuJoCo geom/body xmat is row-major 9-vector; reshape to (3,3)."""
    return xmat_flat9.reshape(3, 3)


def extract_keypoints_from_csv(
    csv_path: Path,
    fps: float = 30.0,
    max_frames: int | None = None,
) -> dict:
    g1_rows = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if g1_rows.ndim == 1:
        g1_rows = g1_rows.reshape(1, -1)
    if max_frames is not None:
        g1_rows = g1_rows[:max_frames]
    T = g1_rows.shape[0]

    # The G1 URDF (loaded via load_g1_model) is fixed-base — body 0 is world,
    # joints take qpos directly with no freejoint. We apply the root pose by
    # transforming body world poses ourselves after FK.
    model = load_g1_model()
    data = mujoco.MjData(model)

    body_id = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    joint_qpos_idx = np.array([
        int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")
        ])
        for n in G1_CSV_JOINT_ORDER
    ], dtype=np.int32)

    # Resolve body indices for each base keypoint (None = pelvis or derived).
    keypoint_body_ids: list[int] = []
    for name, link in G1_KEYPOINT_LINKS:
        if link is None:
            keypoint_body_ids.append(-1)
            continue
        bid = body_id(link)
        if bid < 0:
            raise RuntimeError(f"G1 URDF missing body for {name}: {link}")
        keypoint_body_ids.append(bid)
    left_ankle_bid  = body_id("left_ankle_roll_link")
    right_ankle_bid = body_id("right_ankle_roll_link")
    left_wrist_bid  = body_id("left_wrist_yaw_link")
    right_wrist_bid = body_id("right_wrist_yaw_link")
    torso_bid       = body_id(TORSO_LINK_NAME)
    if torso_bid < 0:
        raise RuntimeError(
            f"G1 URDF missing torso body '{TORSO_LINK_NAME}'. Update TORSO_LINK_NAME."
        )

    positions    = np.zeros((T, N_RETARGET + 3, 3), dtype=np.float32)
    orientations = np.zeros((T, N_RETARGET + 3, 3, 3), dtype=np.float32)
    foot_zs      = np.zeros((T, 2), dtype=np.float32)  # left, right ankle z

    # Pre-compute root rotation matrices in the world frame so we can transform
    # G1's pelvis-relative body poses into world-frame keypoints.
    root_pos_world = g1_rows[:, 0:3].astype(np.float64)
    root_quat_xyzw = g1_rows[:, 3:7].astype(np.float64)

    for t in range(T):
        data.qpos[:] = 0.0
        data.qpos[joint_qpos_idx] = g1_rows[t, 7:]
        mujoco.mj_kinematics(model, data)

        # Build root rotation matrix from xyzw quaternion (rotation only, scaled).
        x, y, z, w = root_quat_xyzw[t]
        n = max(np.sqrt(w * w + x * x + y * y + z * z), 1e-9)
        x, y, z, w = x / n, y / n, z / n, w / n
        R_root = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])

        # Helper: transform a body's pelvis-relative (FK) pose into world.
        def world_pose(bid: int) -> tuple[np.ndarray, np.ndarray]:
            # data.xpos[bid] is relative to G1's URDF root (which is the pelvis)
            # because the URDF is fixed-base.
            p_local = np.array(data.xpos[bid], dtype=np.float64)
            R_local = _xmat_to_rotmat(np.array(data.xmat[bid], dtype=np.float64))
            p_world = R_root @ p_local + root_pos_world[t]
            R_world = R_root @ R_local
            return p_world, R_world

        for i, (kp_name, link) in enumerate(G1_KEYPOINT_LINKS):
            if kp_name == "pelvis":
                p, R = root_pos_world[t], R_root
            elif link is None:
                # Foot keypoint — derive from ankle + local forward offset.
                if "left" in kp_name:
                    p, R = world_pose(left_ankle_bid)
                else:
                    p, R = world_pose(right_ankle_bid)
                p = p + R @ FOOT_LOCAL_OFFSET
            else:
                p, R = world_pose(keypoint_body_ids[i])
            positions[t, i] = p
            orientations[t, i] = R

        # Aux points.
        l_wrist_p, l_wrist_R = world_pose(left_wrist_bid)
        r_wrist_p, r_wrist_R = world_pose(right_wrist_bid)
        torso_p,   torso_R   = world_pose(torso_bid)

        positions[t, 15] = l_wrist_p + l_wrist_R @ LEFT_HAND_AUX_OFFSET
        orientations[t, 15] = l_wrist_R
        positions[t, 16] = r_wrist_p + r_wrist_R @ RIGHT_HAND_AUX_OFFSET
        orientations[t, 16] = r_wrist_R
        positions[t, 17] = torso_p + torso_R @ TORSO_AUX_LOCAL
        orientations[t, 17] = torso_R

        foot_zs[t, 0] = positions[t, 5, 2]   # left_ankle
        foot_zs[t, 1] = positions[t, 6, 2]   # right_ankle

    # Foot contact heuristic from ankle (idx 5/6) and foot (idx 7/8) z-height
    # plus per-frame finite-difference velocity magnitude.
    dt = 1.0 / fps
    def contact_pair(ankle_idx: int, foot_idx: int) -> np.ndarray:
        p_a = positions[:, ankle_idx, :]
        p_f = positions[:, foot_idx, :]
        v_a = np.zeros_like(p_a); v_a[1:] = (p_a[1:] - p_a[:-1]) / dt
        v_f = np.zeros_like(p_f); v_f[1:] = (p_f[1:] - p_f[:-1]) / dt
        contact_a = (p_a[:, 2] < CONTACT_HEIGHT) & (np.linalg.norm(v_a, axis=1) < CONTACT_VEL)
        contact_f = (p_f[:, 2] < CONTACT_HEIGHT) & (np.linalg.norm(v_f, axis=1) < CONTACT_VEL)
        return np.stack([contact_a.astype(np.int64), contact_f.astype(np.int64)], axis=1)

    return {
        "positions":           positions,
        "orientations":        orientations,
        "left_foot_contacts":  contact_pair(5, 7),
        "right_foot_contacts": contact_pair(6, 8),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="LAFAN1 G1 CSV file.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output keypoints .npy path.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Source FPS (LAFAN1 is 30). Used by contact heuristic.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap T to first N frames (useful for testing).")
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 2

    out = extract_keypoints_from_csv(args.csv, fps=args.fps, max_frames=args.max_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)

    pos = out["positions"]
    print(f"wrote {args.out}")
    print(f"  positions:           {pos.shape} dtype={pos.dtype}")
    print(f"  left_foot_contacts:  {out['left_foot_contacts'].shape}  "
          f"frac_contact={out['left_foot_contacts'][:, 0].mean():.2f}")
    print(f"  right_foot_contacts: {out['right_foot_contacts'].shape}  "
          f"frac_contact={out['right_foot_contacts'][:, 0].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
