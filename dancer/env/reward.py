"""DeepMimic-style tracking reward terms for Asimov motion imitation.

Every error term is computed on the **23 actuated joints only**. The 4
passive joints (toes + neck) cannot be driven by the real robot, so
penalising the policy for failing to match their (free / springy) dynamics
would teach it the wrong thing. The reward operates on
`joint_q[..., actuated_idx]` against `ref_joint_q[..., actuated_idx]` —
likewise for velocities — exactly as the project memory dictates.

Per-frame reward (per env):

    r_jp = exp(-2.0  * mean_sq(q_act      - ref_q_act))                  # joint pos
    r_jv = exp(-0.1  * mean_sq(qd_act     - ref_qd_act))                 # joint vel
    r_rh = exp(-50.0 * (root_z - ref_root_z) ** 2)                       # root height
    r_rp = exp(-20.0 * sum_sq(root_xy - ref_root_xy))                    # planar pos
    r_rq = exp(-2.0  * quat_angle(root_q, ref_root_q) ** 2)              # orientation

    tracking = w_jp r_jp + w_jv r_jv + w_rh r_rh + w_rp r_rp + w_rq r_rq

Plus constants:
    fallen_penalty = -w_fall * fallen.float()       (typ. w_fall=1.0)
    action_penalty = -w_action * mean(action ** 2)  (typ. w_action=0.005)
    alive_bonus    = +w_alive * (~fallen).float()   (typ. w_alive=0.05)

`fallen` is also returned for the env to use as a termination signal.

Quaternion convention is xyzw end-to-end (project memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Math helpers (all batched over the leading dim N = num_envs)
# ---------------------------------------------------------------------------

def quat_angle_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Shortest-path angle between two unit quaternions (xyzw).

    Returns (...,) radians in [0, π].
    """
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def body_up_z(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """World-frame z-component of the body's local +Z axis (i.e. R[2, 2]).

    Equals 1 when the body is upright, 0 when its z-axis points horizontally,
    -1 when fully inverted. Used as the "is fallen" orientation check.
    """
    qx = quat_xyzw[..., 0]
    qy = quat_xyzw[..., 1]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def is_fallen(
    base_pos: torch.Tensor,
    base_quat_xyzw: torch.Tensor,
    *,
    z_fall: float,
    up_dot_min: float,
) -> torch.Tensor:
    """Per-env fallen-state bool: pelvis too low OR robot tipped over."""
    low = base_pos[..., 2] < z_fall
    tipped = body_up_z(base_quat_xyzw) < up_dot_min
    return low | tipped


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    """Reward term weights. Defaults match `dancer/configs/reward/deepmimic.yaml`."""
    w_jp: float = 0.50
    w_jv: float = 0.10
    w_rh: float = 0.20
    w_rp: float = 0.05
    w_rq: float = 0.15
    w_fall: float = 1.0
    w_action: float = 0.005
    w_alive: float = 0.05
    # Sharpness of each exp() tracking term: r = exp(-s * err). Larger = the
    # reward keeps rewarding tighter tracking instead of saturating near 1.
    # Defaults reproduce the original hard-coded values.
    s_jp: float = 2.0
    s_jv: float = 0.1
    s_rh: float = 50.0
    s_rp: float = 20.0
    s_rq: float = 2.0
    # If True, the joint-position reward is the MEAN of per-joint exp() terms
    # (mean_j exp(-s_jp*err_j^2)) instead of exp(-s_jp*mean_j err_j^2). The
    # per-joint form does not let a few well-tracked joints mask badly-tracked
    # ones — each joint must track to earn its share. Critical for the lateral
    # leg joints (hip roll/yaw) that the averaged form lets the policy ignore.
    jp_per_joint: bool = False
    # Optional per-actuated-joint weights (length n_actuated, in actuated-name
    # order) for the per-joint reward — lets us emphasise the balance-critical
    # lateral leg joints (hip roll/yaw, waist) the policy tends to damp. Empty
    # tuple = uniform. Only used when jp_per_joint is True.
    jp_weights: tuple = ()


def compute_reward(
    *,
    # Robot state (N, ...)
    joint_q: torch.Tensor,           # (N, n_joints)
    joint_qd: torch.Tensor,          # (N, n_joints)
    base_pos: torch.Tensor,          # (N, 3)
    base_quat_xyzw: torch.Tensor,    # (N, 4)
    # Reference state (N, ...) at the env's current phase index
    ref_joint_q: torch.Tensor,
    ref_joint_qd: torch.Tensor,
    ref_base_pos: torch.Tensor,
    ref_base_quat_xyzw: torch.Tensor,
    # Action that produced this state (N, n_actuated)
    action: torch.Tensor,
    # Which joints are actuated (slice into n_joints axis)
    actuated_idx: torch.Tensor,      # (n_actuated,) long
    # Hyperparameters
    weights: RewardWeights,
    fallen: torch.Tensor,            # (N,) bool — computed by caller via is_fallen()
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns (reward (N,), info dict of per-term tensors for logging)."""
    q_act     = joint_q[..., actuated_idx]
    qd_act    = joint_qd[..., actuated_idx]
    ref_q_act  = ref_joint_q[..., actuated_idx]
    ref_qd_act = ref_joint_qd[..., actuated_idx]

    # --- tracking terms (each in (0, 1]) ---
    sq_jp = (q_act - ref_q_act) ** 2
    e_jp = sq_jp.mean(dim=-1)
    if weights.jp_per_joint:
        per_joint = torch.exp(-weights.s_jp * sq_jp)             # (N, n_actuated)
        if weights.jp_weights:
            w = torch.as_tensor(weights.jp_weights, dtype=per_joint.dtype,
                                 device=per_joint.device)
            r_jp = (per_joint * w).sum(dim=-1) / w.sum()
        else:
            r_jp = per_joint.mean(dim=-1)
    else:
        r_jp = torch.exp(-weights.s_jp * e_jp)

    e_jv = ((qd_act - ref_qd_act) ** 2).mean(dim=-1)
    r_jv = torch.exp(-weights.s_jv * e_jv)

    dz = base_pos[..., 2] - ref_base_pos[..., 2]
    r_rh = torch.exp(-weights.s_rh * dz * dz)

    dxy = base_pos[..., :2] - ref_base_pos[..., :2]
    e_rp = (dxy * dxy).sum(dim=-1)
    r_rp = torch.exp(-weights.s_rp * e_rp)

    ang = quat_angle_distance(base_quat_xyzw, ref_base_quat_xyzw)
    r_rq = torch.exp(-weights.s_rq * ang * ang)

    tracking = (
        weights.w_jp * r_jp
        + weights.w_jv * r_jv
        + weights.w_rh * r_rh
        + weights.w_rp * r_rp
        + weights.w_rq * r_rq
    )

    # --- constants outside the multiplicative tracking term ---
    fallen_f = fallen.float()
    alive_f = 1.0 - fallen_f
    fallen_penalty = -weights.w_fall * fallen_f
    alive_bonus = weights.w_alive * alive_f
    action_penalty = -weights.w_action * (action ** 2).mean(dim=-1)

    reward = tracking + fallen_penalty + action_penalty + alive_bonus

    info = {
        "r_jp": r_jp,
        "r_jv": r_jv,
        "r_rh": r_rh,
        "r_rp": r_rp,
        "r_rq": r_rq,
        "tracking": tracking,
        "fallen_penalty": fallen_penalty,
        "action_penalty": action_penalty,
        "alive_bonus": alive_bonus,
        "fallen": fallen_f,
        "joint_pos_err_rad": e_jp.sqrt(),  # RMS of (q - ref_q) over actuated dims
    }
    return reward, info
