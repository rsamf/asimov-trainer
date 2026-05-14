"""PyRoKi-based retargeting from human keypoint motion to Asimov-v1.

Mirrors the structure of EXAMPLE_to_g1_from_keypoints.py (which targets the
Unitree G1) but points at the Asimov-v1 URDF and remaps the human-to-robot
link correspondences for Asimov's joint topology.

Differences from the G1 reference:
  * URDF: data/robot/asimov-v1/xmls/asimov.urdf
  * Pelvis link: `pelvis_link` (not `pelvis_contour_link`).
  * Foot link: `*_toe_link` (Asimov's distal foot segment).
  * Torso aux: anchored on `waist_yaw_link` (Asimov has no `torso_link`).
  * `joints_to_move_less`: Asimov's passive toes & neck DOFs (toe & neck
    joints exist in the URDF but aren't actuated in the MJCF; the optimizer
    should leave them near rest). G1's waist_roll/wrist_pitch entries are
    not applicable — Asimov has only `waist_yaw_joint` (worth using) and a
    single `*_wrist_yaw_joint` per wrist (also worth using).

Input keypoint `.npy` schema (matches ProtoMotions' extractor):
    {
      "positions":           (T, 18, 3) float32,
      "orientations":        (T, 18, 3, 3) float32,
      "left_foot_contacts":  (T, 2) int,   # ankle, toebase
      "right_foot_contacts": (T, 2) int,
    }

Output `.npz` per motion:
    base_frame_pos:   (T, 3),
    base_frame_wxyz:  (T, 4),
    joint_angles:     (T, n_actuated) — in `robot.joints.actuated_names` order.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy


# ---------------------------------------------------------------------------
# Robot-specific constants
# ---------------------------------------------------------------------------

ASIMOV_URDF = Path("data/robot/asimov-v1/xmls/asimov.urdf")
# yourdfpy resolves relative <mesh filename="../assets/meshes/..."/> against the
# directory passed as mesh_dir, so we pass the URDF's own directory.
ASIMOV_MESH_DIR = ASIMOV_URDF.parent

N_RETARGET = 15
N_AUX = 3

# Source keypoint name -> Asimov URDF link name. Order matters; index in this
# list is also the index inside the keypoint `positions[t]` array.
HUMAN_TO_ASIMOV_LINKS: list[tuple[str, str]] = [
    ("pelvis",         "pelvis_link"),
    ("left_hip",       "left_hip_pitch_link"),
    ("right_hip",      "right_hip_pitch_link"),
    ("left_knee",      "left_knee_link"),
    ("right_knee",     "right_knee_link"),
    ("left_ankle",     "left_ankle_roll_link"),
    ("right_ankle",    "right_ankle_roll_link"),
    ("left_foot",      "left_toe_link"),
    ("right_foot",     "right_toe_link"),
    ("left_shoulder",  "left_shoulder_pitch_link"),
    ("right_shoulder", "right_shoulder_pitch_link"),
    ("left_elbow",     "left_elbow_link"),
    ("right_elbow",    "right_elbow_link"),
    ("left_wrist",     "left_wrist_yaw_link"),
    ("right_wrist",    "right_wrist_yaw_link"),
]
assert len(HUMAN_TO_ASIMOV_LINKS) == N_RETARGET

# Bones whose endpoints we additionally constrain by relative-position cost
# (the "local alignment" / direct-pairs term).
DIRECT_PAIRS: list[tuple[str, str, float]] = [
    ("left_shoulder",  "left_elbow",     1.0),
    ("right_shoulder", "right_elbow",    1.0),
    ("left_elbow",     "left_wrist",     1.0),
    ("right_elbow",    "right_wrist",    1.0),
    ("left_hip",       "left_knee",      1.0),
    ("right_hip",      "right_knee",     1.0),
    ("left_knee",      "left_ankle",     1.0),
    ("right_knee",     "right_ankle",    1.0),
    ("left_ankle",     "left_foot",      1.0),
    ("right_ankle",    "right_foot",     1.0),
]

# Asimov joints to keep near rest. Toes + neck are passive springs in the MJCF
# (no actuators) and shouldn't carry retargeted motion. The wrist_yaw is the
# closest analog to G1's wrist_pitch (forearm twist) — penalising it leaves
# wrist orientation gentle rather than spinning to track noisy keypoints.
ASIMOV_JOINTS_TO_MOVE_LESS: list[str] = [
    "left_toe_joint",
    "right_toe_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Aux link offsets (in each link's local frame). Tuned for Asimov geometry.
LEFT_HAND_AUX_OFFSET  = onp.array([0.0, 0.0, -0.10])  # wrist link extends to ~z=-0.08
RIGHT_HAND_AUX_OFFSET = onp.array([0.0, 0.0, -0.10])
TORSO_AUX_OFFSET      = onp.array([0.0, 0.0,  0.20])  # ~mid-chest above waist link
TORSO_LINK_NAME       = "waist_yaw_link"


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    root_smoothness: float
    joint_smoothness: float
    self_collision: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float


# ---------------------------------------------------------------------------
# Module-level state populated by main() before solve_retargeting runs.
# ---------------------------------------------------------------------------

ASIMOV_LINK_NAMES: list[str] | None = None
human_retarget_names: list[str] | None = None
asimov_joint_retarget_indices: jnp.ndarray | None = None
TORSO_LINK_INDEX: int | None = None


def _resolve_indices() -> tuple[list[str], jnp.ndarray]:
    names = []
    idxs = []
    for human, link in HUMAN_TO_ASIMOV_LINKS:
        names.append(human)
        idxs.append(ASIMOV_LINK_NAMES.index(link))
    return names, jnp.array(idxs)


# ---------------------------------------------------------------------------
# Motion data loading. Identical in spirit to the G1 example.
# ---------------------------------------------------------------------------

def load_motion_data(motion_path, source_type, subsample_factor, target_raw_frames):
    print(f"Loading motion from: {motion_path}")
    motion_data = onp.load(motion_path, allow_pickle=True).item()

    target_subsampled_frames = len(list(range(0, target_raw_frames, subsample_factor)))

    raw_positions          = motion_data["positions"]
    raw_orientations       = motion_data["orientations"]
    raw_left_foot_contacts  = motion_data["left_foot_contacts"]   # [T, 2]
    raw_right_foot_contacts = motion_data["right_foot_contacts"]  # [T, 2]
    original_raw_frames = raw_positions.shape[0]

    print(f"  original frames: {original_raw_frames}")
    assert original_raw_frames > 0
    original_subsampled = raw_positions[::subsample_factor].shape[0]
    num_timesteps = min(original_subsampled, target_subsampled_frames)
    print(f"  subsampled frames used: {num_timesteps} (capped at {target_subsampled_frames})")

    # Pad/trim to fixed solver buffer length.
    if original_raw_frames >= target_raw_frames:
        proc_pos    = raw_positions[:target_raw_frames]
        proc_orient = raw_orientations[:target_raw_frames]
        proc_lc     = raw_left_foot_contacts[:target_raw_frames]
        proc_rc     = raw_right_foot_contacts[:target_raw_frames]
    else:
        pad = target_raw_frames - original_raw_frames
        proc_pos    = onp.concatenate((raw_positions,    onp.repeat(raw_positions[-1:],    pad, axis=0)), axis=0)
        proc_orient = onp.concatenate((raw_orientations, onp.repeat(raw_orientations[-1:], pad, axis=0)), axis=0)
        proc_lc     = onp.concatenate((raw_left_foot_contacts,  onp.repeat(raw_left_foot_contacts[-1:],  pad, axis=0)), axis=0)
        proc_rc     = onp.concatenate((raw_right_foot_contacts, onp.repeat(raw_right_foot_contacts[-1:], pad, axis=0)), axis=0)

    # Cross-fade contacts before subsampling.
    left_avg  = onp.mean(proc_lc.astype(float), axis=1)[:, None]
    right_avg = onp.mean(proc_rc.astype(float), axis=1)[:, None]
    window = 5
    def crossfade(flags):
        out = onp.zeros_like(flags)
        for i in range(len(flags)):
            lo = max(0, i - window // 2)
            hi = min(len(flags), i + window // 2 + 1)
            out[i] = onp.mean(flags[lo:hi])
        return out
    left_smooth  = crossfade(left_avg)
    right_smooth = crossfade(right_avg)

    keypoints = proc_pos[::subsample_factor]

    # SMPL/RIG humans are ~10-20% taller than Asimov; scale local keypoints
    # toward the robot's size before the optimizer sees them.
    root = keypoints[:, 0, :]
    local = keypoints - root[:, None, :]
    if source_type == "smpl":
        lower_scale = onp.array([0.85, 0.85, 0.80])
        upper_scale = onp.array([0.85, 0.85, 0.75])
        root_scale  = onp.array([0.85, 0.85, 0.80])
    elif source_type == "rigv1":
        lower_scale = onp.array([0.80, 0.80, 0.75])
        upper_scale = onp.array([0.80, 0.80, 0.70])
        root_scale  = onp.array([0.80, 0.80, 0.75])
    else:
        raise ValueError(f"unknown source_type: {source_type}")

    lower_local = local[:, 1:9, :] * lower_scale
    upper_local = local[:, 9:N_RETARGET + N_AUX, :] * upper_scale
    local = onp.concatenate([lower_local, upper_local], axis=1)
    root = root * root_scale
    keypoints = onp.concatenate([root[:, None, :], root[:, None, :] + local], axis=1)

    orientations = proc_orient[::subsample_factor]
    lc = left_smooth[::subsample_factor]
    rc = right_smooth[::subsample_factor]

    expected_pos    = (target_subsampled_frames, N_RETARGET + N_AUX, 3)
    expected_orient = (target_subsampled_frames, N_RETARGET + N_AUX, 3, 3)
    expected_lc     = (target_subsampled_frames, 1)
    assert keypoints.shape    == expected_pos,    f"positions {keypoints.shape} != {expected_pos}"
    assert orientations.shape == expected_orient, f"orientations {orientations.shape} != {expected_orient}"
    assert lc.shape == expected_lc and rc.shape == expected_lc, \
        f"contacts {lc.shape}/{rc.shape} != {expected_lc}"

    return keypoints, orientations, lc, rc, num_timesteps


# ---------------------------------------------------------------------------
# Cost factories — these are structurally identical to the G1 reference but
# refer to ASIMOV_LINK_NAMES / asimov_joint_retarget_indices via globals so
# the JIT-compiled solver sees them as JAX-static through closure.
# ---------------------------------------------------------------------------

@jaxls.Cost.create_factory
def joint_vel_limit_cost(
    var_values: jaxls.VarValues,
    var_joints_curr: jaxls.Var[jnp.ndarray],
    var_joints_prev: jaxls.Var[jnp.ndarray],
    max_vel: float,
    dt: float,
    weight: float,
) -> jax.Array:
    joints_curr = var_values[var_joints_curr]
    joints_prev = var_values[var_joints_prev]
    joint_vel = (joints_curr - joints_prev) / dt
    excess = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)
    return excess.flatten() * weight


@jaxls.Cost.create_factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    asimov_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    T_world_root_curr = var_values[var_Ts_world_root_curr]
    T_world_root_prev = var_values[var_Ts_world_root_prev]
    cfg_curr = var_values[var_robot_cfg_curr]
    cfg_prev = var_values[var_robot_cfg_prev]

    T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_curr))
    T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_prev))
    T_world_link_curr = T_world_root_curr @ T_root_link_curr
    T_world_link_prev = T_world_root_prev @ T_root_link_prev

    left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx = foot_indices
    left_ankle_robot  = asimov_joint_retarget_indices[left_ankle_idx]
    right_ankle_robot = asimov_joint_retarget_indices[right_ankle_idx]
    left_foot_robot   = asimov_joint_retarget_indices[left_foot_idx]
    right_foot_robot  = asimov_joint_retarget_indices[right_foot_idx]

    pos_curr = T_world_link_curr.translation()
    pos_prev = T_world_link_prev.translation()

    la_curr, ra_curr = pos_curr[left_ankle_robot], pos_curr[right_ankle_robot]
    lf_curr, rf_curr = pos_curr[left_foot_robot],  pos_curr[right_foot_robot]
    la_prev, ra_prev = pos_prev[left_ankle_robot], pos_prev[right_ankle_robot]
    lf_prev, rf_prev = pos_prev[left_foot_robot],  pos_prev[right_foot_robot]

    la_vel, ra_vel = la_curr - la_prev, ra_curr - ra_prev
    lf_vel, rf_vel = lf_curr - lf_prev, rf_curr - rf_prev

    l_z_diff = la_curr[2] - lf_curr[2]
    r_z_diff = ra_curr[2] - rf_curr[2]

    lw = left_foot_contact[0]
    rw = right_foot_contact[0]

    return (
        jnp.concatenate([
            (lw * la_vel).flatten(),
            (rw * ra_vel).flatten(),
            (lw * lf_vel).flatten(),
            (rw * rf_vel).flatten(),
            jnp.array([lw * l_z_diff]),
            jnp.array([rw * r_z_diff]),
        ]) * weight
    )


@jaxls.Cost.create_factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    asimov_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    T_world_root = var_values[var_Ts_world_root]
    cfg = var_values[var_robot_cfg]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
    T_world_link = T_world_root @ T_root_link

    left_ankle_idx, right_ankle_idx, _, _ = foot_indices
    la_robot = asimov_joint_retarget_indices[left_ankle_idx]
    ra_robot = asimov_joint_retarget_indices[right_ankle_idx]

    l_ori = T_world_link.rotation().as_matrix()[la_robot]
    r_ori = T_world_link.rotation().as_matrix()[ra_robot]

    lw = left_foot_contact[0]
    rw = right_foot_contact[0]

    l_res = lw * (l_ori[2, 2] - 1.0)
    r_res = rw * (r_ori[2, 2] - 1.0)
    return jnp.concatenate([jnp.array([l_res]), jnp.array([r_res])]) * weight


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    target_orientations: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    asimov_joint_retarget_indices: jnp.ndarray,
    asimov_retarget_mask: jnp.ndarray,
    weights: RetargetingWeights,
    subsample_factor: int = 1,
    input_fps: float = 30.0,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the keypoint-to-Asimov retargeting problem."""
    n_retarget = len(asimov_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Asimov joints we want to hold near rest.
    joints_to_move_less = jnp.array([
        robot.joints.actuated_names.index(n) for n in ASIMOV_JOINTS_TO_MOVE_LESS
        if n in robot.joints.actuated_names
    ])

    foot_indices = jnp.array([
        human_retarget_names.index("left_ankle"),
        human_retarget_names.index("right_ankle"),
        human_retarget_names.index("left_foot"),
        human_retarget_names.index("right_foot"),
    ])

    class JointsScaleVarAsimov(
        jaxls.Var[jax.Array],
        default_factory=lambda: jnp.ones((n_retarget, n_retarget)),
    ): ...

    var_joints        = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale  = JointsScaleVarAsimov(jnp.zeros(timesteps))

    # Initialize root poses from source root keypoint at each frame.
    root_init_se3 = []
    for t in range(timesteps):
        root_pos_t = target_keypoints[t, 0, :]
        root_rot_t = target_orientations[t, 0, :, :]
        root_init_se3.append(
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(root_rot_t), root_pos_t
            )
        )
    root_init_values = jaxlie.SE3(jnp.stack([se3.wxyz_xyz for se3 in root_init_se3]))

    # ---- Cost: local bones alignment (relative positions/angles).
    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: JointsScaleVarAsimov,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        target_pos = keypoints[:N_RETARGET, :]
        robot_pos = T_world_link.translation()[jnp.array(asimov_joint_retarget_indices)]

        delta_target = target_pos[:, None] - target_pos[None, :]
        delta_robot  = robot_pos[:,  None] - robot_pos[None, :]

        position_scale = var_values[var_joints_scale][..., None]
        residual_position_delta = (
            (delta_target - delta_robot * position_scale)
            * (1 - jnp.eye(delta_target.shape[0])[..., None])
            * asimov_retarget_mask[..., None]
        )

        delta_target_n = delta_target / jnp.linalg.norm(delta_target + 1e-6, axis=-1, keepdims=True)
        delta_robot_n  = delta_robot  / jnp.linalg.norm(delta_robot  + 1e-6, axis=-1, keepdims=True)
        residual_angle_delta = 1 - (delta_target_n * delta_robot_n).sum(axis=-1)
        residual_angle_delta = (
            residual_angle_delta * (1 - jnp.eye(residual_angle_delta.shape[0])) * asimov_retarget_mask
        )

        return jnp.concatenate([
            residual_position_delta.flatten(),
            residual_angle_delta.flatten(),
        ]) * weights["local_alignment"]

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: JointsScaleVarAsimov,
    ) -> jax.Array:
        s = var_values[var_joints_scale]
        res_0 = (s - 1.0).flatten() * 1.0
        res_1 = (s - s.T).flatten() * 100.0
        res_2 = jnp.clip(-s, min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        T_world_root = var_values[var_Ts_world_root]
        cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[asimov_joint_retarget_indices]

        # Hand aux points (one per wrist link, offset down the forearm axis).
        l_wrist = human_retarget_names.index("left_wrist")
        l_wrist_idx = asimov_joint_retarget_indices[l_wrist]
        l_wrist_pos = T_world_link.translation()[l_wrist_idx]
        l_wrist_R   = T_world_link.rotation().as_matrix()[l_wrist_idx]
        l_hand_aux  = l_wrist_pos + l_wrist_R @ jnp.asarray(LEFT_HAND_AUX_OFFSET)

        r_wrist = human_retarget_names.index("right_wrist")
        r_wrist_idx = asimov_joint_retarget_indices[r_wrist]
        r_wrist_pos = T_world_link.translation()[r_wrist_idx]
        r_wrist_R   = T_world_link.rotation().as_matrix()[r_wrist_idx]
        r_hand_aux  = r_wrist_pos + r_wrist_R @ jnp.asarray(RIGHT_HAND_AUX_OFFSET)

        # Torso aux: anchored on Asimov's waist_yaw_link (no torso_link in MJCF).
        torso_pos = T_world_link.translation()[TORSO_LINK_INDEX]
        torso_R   = T_world_link.rotation().as_matrix()[TORSO_LINK_INDEX]
        torso_aux = torso_pos + torso_R @ jnp.asarray(TORSO_AUX_OFFSET)

        link_pos_with_aux = jnp.concatenate([
            link_pos, l_hand_aux[None, :], r_hand_aux[None, :], torso_aux[None, :],
        ], axis=0)

        kp = keypoints
        # Down-weight noisy aux + elbow channels.
        kp = kp.at[-2, :].set(kp[-2, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-2, :].set(link_pos_with_aux[-2, :] / 4.0)
        kp = kp.at[-3, :].set(kp[-3, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-3, :].set(link_pos_with_aux[-3, :] / 4.0)
        kp = kp.at[-6, :].set(kp[-6, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-6, :].set(link_pos_with_aux[-6, :] / 4.0)
        kp = kp.at[-7, :].set(kp[-7, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-7, :].set(link_pos_with_aux[-7, :] / 4.0)

        return (link_pos_with_aux - kp).flatten() * weights["global_alignment"]

    @jaxls.Cost.create_factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    costs: list[jaxls.Cost] = [
        retargeting_cost(var_Ts_world_root, var_joints, var_joints_scale, target_keypoints),
        scale_regularization(var_joints_scale),
        pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), var_joints, 100.0),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        pc_alignment_cost(var_Ts_world_root, var_joints, target_keypoints),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(var_joints.default_factory().shape, 0.02)
            .at[joints_to_move_less].set(weights["joint_rest_penalty"])[None],
        ),
        joint_vel_limit_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            20.0,
            subsample_factor / input_fps,
            weights["joint_vel_limit"],
        ),
    ]

    for t in range(1, timesteps):
        costs.append(foot_contact_cost(
            jaxls.SE3Var(t), jaxls.SE3Var(t - 1),
            robot.joint_var_cls(t), robot.joint_var_cls(t - 1),
            robot,
            left_foot_contact[t], right_foot_contact[t],
            asimov_joint_retarget_indices, foot_indices,
            weights["foot_contact"],
        ))
    for t in range(timesteps):
        costs.append(foot_tilt_cost(
            jaxls.SE3Var(t), robot.joint_var_cls(t), robot,
            left_foot_contact[t], right_foot_contact[t],
            asimov_joint_retarget_indices, foot_indices,
            weights["foot_tilt"],
        ))

    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_joints_scale]
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([
                var_joints,
                var_Ts_world_root.with_value(root_init_values),
                var_joints_scale,
            ]),
            termination=jaxls.TerminationConfig(max_iterations=800),
        )
    )

    return solution[var_Ts_world_root], solution[var_joints]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retarget keypoint motions to Asimov-v1 via PyRoKi.")
    parser.add_argument("--keypoints-folder-path", type=str, required=True,
                        help="Directory of keypoint .npy files (one motion per file).")
    parser.add_argument("--output-dir", type=str, default="./data/motions/asimov-v1-pyroki",
                        help="Where to write retargeted .npz outputs.")
    parser.add_argument("--urdf-path", type=str, default=str(ASIMOV_URDF))
    parser.add_argument("--mesh-dir", type=str, default=str(ASIMOV_MESH_DIR))
    parser.add_argument("--source-type", type=str, default="smpl",
                        choices=["smpl", "rigv1"],
                        help="Source skeleton type (controls scale heuristics).")
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--target-raw-frames", type=int, default=450)
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Run headless and write all retargeted motions to --output-dir.")
    args = parser.parse_args()

    keypoint_paths = sorted(glob.glob(os.path.join(args.keypoints_folder_path, "*.npy")))
    if not keypoint_paths:
        print(f"no .npy files in {args.keypoints_folder_path}")
        return 2

    urdf = yourdfpy.URDF.load(args.urdf_path, mesh_dir=args.mesh_dir)
    robot = pk.Robot.from_urdf(urdf)

    global ASIMOV_LINK_NAMES, human_retarget_names, asimov_joint_retarget_indices, TORSO_LINK_INDEX
    ASIMOV_LINK_NAMES = list(robot.links.names)
    human_retarget_names, asimov_joint_retarget_indices = _resolve_indices()
    TORSO_LINK_INDEX = ASIMOV_LINK_NAMES.index(TORSO_LINK_NAME)

    n_retarget = len(asimov_joint_retarget_indices)
    asimov_retarget_mask = jnp.zeros((n_retarget, n_retarget))
    for a, b, w in DIRECT_PAIRS:
        ia = human_retarget_names.index(a)
        ib = human_retarget_names.index(b)
        asimov_retarget_mask = asimov_retarget_mask.at[ia, ib].set(w)
        asimov_retarget_mask = asimov_retarget_mask.at[ib, ia].set(w)

    weights = RetargetingWeights(
        local_alignment=1.0,
        global_alignment=4.0,
        root_smoothness=1.0,
        joint_smoothness=4.0,
        self_collision=0.0,
        joint_rest_penalty=1.0,
        joint_vel_limit=50.0,
        foot_contact=30.0,
        foot_tilt=1.0,
    )

    if args.visualize:
        return _run_visualize(
            args, robot, urdf, keypoint_paths,
            asimov_joint_retarget_indices, asimov_retarget_mask, weights,
        )

    print(f"Retargeting {len(keypoint_paths)} motions to Asimov  →  {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    for i, motion_path in enumerate(keypoint_paths):
        print(f"[{i+1}/{len(keypoint_paths)}] {os.path.basename(motion_path)}")
        stem = Path(motion_path).stem
        out_path = Path(args.output_dir) / f"{stem}_retargeted.npz"
        if args.skip_existing and out_path.exists():
            print(f"  exists, skipping  ({out_path.name})")
            continue

        keypoints, orientations, lc, rc, num_timesteps = load_motion_data(
            motion_path, args.source_type, args.subsample_factor, args.target_raw_frames,
        )
        t0 = time.perf_counter()
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=keypoints,
            target_orientations=orientations,
            left_foot_contact=lc,
            right_foot_contact=rc,
            asimov_joint_retarget_indices=asimov_joint_retarget_indices,
            asimov_retarget_mask=asimov_retarget_mask,
            weights=weights,
            subsample_factor=args.subsample_factor,
            input_fps=args.input_fps,
        )
        elapsed = time.perf_counter() - t0
        print(f"  solved in {elapsed:5.1f}s ({num_timesteps} kept frames)")

        results = {
            "base_frame_pos":  onp.asarray(Ts_world_root.wxyz_xyz[:num_timesteps, 4:]),
            "base_frame_wxyz": onp.asarray(Ts_world_root.wxyz_xyz[:num_timesteps, :4]),
            "joint_angles":    onp.asarray(joints[:num_timesteps]),
        }
        onp.savez_compressed(out_path, **results)
        print(f"  wrote {out_path}")
    return 0


def _run_visualize(
    args, robot, urdf, keypoint_paths,
    asimov_joint_retarget_indices, asimov_retarget_mask, weights,
):
    import viser
    from viser.extras import ViserUrdf

    current_index = [0]
    keypoints, orientations, lc, rc, num_timesteps = load_motion_data(
        keypoint_paths[0], args.source_type, args.subsample_factor, args.target_raw_frames,
    )

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider(
        "timestep", 0, max(0, num_timesteps - 1), 1, 0
    )

    weight_tuner = pk.viewer.WeightTuner(server, weights)  # type: ignore

    state = {"Ts_world_root": None, "joints": None,
             "keypoints": keypoints, "orientations": orientations,
             "lc": lc, "rc": rc, "num_timesteps": num_timesteps}

    def regenerate():
        gen_button.disabled = True
        retarget_next_button.disabled = True
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=state["keypoints"],
            target_orientations=state["orientations"],
            left_foot_contact=state["lc"],
            right_foot_contact=state["rc"],
            asimov_joint_retarget_indices=asimov_joint_retarget_indices,
            asimov_retarget_mask=asimov_retarget_mask,
            weights=weight_tuner.get_weights(),  # type: ignore
            subsample_factor=args.subsample_factor,
            input_fps=args.input_fps,
        )
        state["Ts_world_root"] = Ts_world_root
        state["joints"] = joints
        gen_button.disabled = False
        retarget_next_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: regenerate())

    def on_next(_):
        current_index[0] = (current_index[0] + 1) % len(keypoint_paths)
        kp, ori, llc, rrc, nt = load_motion_data(
            keypoint_paths[current_index[0]], args.source_type,
            args.subsample_factor, args.target_raw_frames,
        )
        state["keypoints"] = kp
        state["orientations"] = ori
        state["lc"] = llc
        state["rc"] = rrc
        state["num_timesteps"] = nt
        timestep_slider.max = max(0, nt - 1)
        timestep_slider.value = 0
        regenerate()

    retarget_next_button = server.gui.add_button("Retarget Next")
    retarget_next_button.on_click(on_next)

    regenerate()

    while True:
        with server.atomic():
            if playing.value and state["num_timesteps"] > 0:
                timestep_slider.value = (timestep_slider.value + 1) % state["num_timesteps"]
            tstep = timestep_slider.value
        try:
            base_frame.wxyz = onp.array(state["Ts_world_root"].wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(state["Ts_world_root"].wxyz_xyz[tstep][4:])
            urdf_vis.update_cfg(onp.array(state["joints"][tstep]))
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(state["keypoints"][tstep]),
                onp.array((0, 0, 255))[None].repeat(state["keypoints"].shape[1], axis=0),
                point_size=0.01,
            )
        except Exception:
            pass
        time.sleep(args.subsample_factor / args.input_fps)


if __name__ == "__main__":
    raise SystemExit(main())
