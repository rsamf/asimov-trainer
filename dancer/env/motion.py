"""Reference motion loader for Asimov dance imitation.

Loads the retargeted .npz (30 fps source, URDF joint order), pre-upsamples
to the physics rate at startup using SLERP for orientations and linear
interpolation for translations / joint angles, and computes finite-difference
velocities at the upsampled rate. Resets snap to integer indices in this
upsampled array — no per-step interpolation work at training time.

The reference covers 27 joints in URDF order; the 23 actuated joints (per the
Asimov MJCF actuator list) are exposed via `actuated_idx`. Reward and target
observation must always slice through `actuated_idx`; the 4 passive joints
(toes + neck) are never used as targets because the real robot cannot drive
them.

Quaternion convention: **xyzw** (vector-scalar) end-to-end. The source `.npz`
uses `base_frame_wxyz` (Pyroki output is wxyz); we convert at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mujoco
import numpy as np
import torch
import yourdfpy

# Canonical 23-joint actuated list for Asimov-v1, in MJCF actuator order. The
# Asimov MJCF intentionally omits the `<actuator>` block (actuators are added
# programmatically in deployment), so we read the list from retargeting's
# shared name table.
from retargeting.joint_map import ASIMOV_ACTUATED_JOINT_NAMES


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw)
# ---------------------------------------------------------------------------

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (xyzw). Shape preserving."""
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b for xyzw quaternions."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([x, y, z, w], dim=-1)


def quat_log_xyz(q: torch.Tensor) -> torch.Tensor:
    """Logarithm of a unit quaternion (xyzw) → 3-vector (the xyz part of log q).

    For q encoding rotation by angle θ about axis u: 2 * log(q) = θ * u, so this
    function returns (θ/2) * u. Combine with a (2/dt) factor for angular velocity.
    """
    xyz = q[..., :3]
    w = q[..., 3]
    norm_xyz = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(norm_xyz.squeeze(-1), w)              # in [0, π]
    # Map (π/2, π] half-angles into (-π/2, 0] so the resulting full angle is in (-π, π].
    half_angle = torch.where(
        half_angle > torch.pi / 2.0, half_angle - torch.pi, half_angle
    )
    safe_norm = norm_xyz.clamp(min=1e-9)
    return half_angle.unsqueeze(-1) * (xyz / safe_norm)


def slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation. q0, q1: (..., 4) xyzw. alpha: (...,) in [0, 1]."""
    # Take the shortest-path rotation by flipping q1 when its dot with q0 is negative.
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot)                                        # (..., 1)
    sin_theta = torch.sin(theta)
    a = alpha.unsqueeze(-1)
    small = sin_theta < 1e-6
    w0 = torch.where(small, 1.0 - a, torch.sin((1.0 - a) * theta) / sin_theta.clamp(min=1e-9))
    w1 = torch.where(small, a,       torch.sin(a * theta)         / sin_theta.clamp(min=1e-9))
    out = w0 * q0 + w1 * q1
    return out / torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# MotionRef
# ---------------------------------------------------------------------------

@dataclass
class MotionRef:
    src_fps: int                            # source motion FPS (LAFAN1 = 30)
    physics_fps: int                        # = src_fps * upsample
    upsample: int                           # integer upsample factor
    T_src: int                              # raw frames in the source npz
    T_up: int                               # = (T_src - 1) * upsample + 1
    n_joints: int                           # all hinges in URDF (27 for Asimov)
    n_actuated: int                         # 23 for Asimov
    device: torch.device

    # All shapes (T_up, ...); on `device`.
    base_pos: torch.Tensor                  # (T_up, 3)
    base_quat: torch.Tensor                 # (T_up, 4) xyzw
    joint_q: torch.Tensor                   # (T_up, n_joints)
    joint_qd: torch.Tensor                  # (T_up, n_joints)
    base_lin_vel: torch.Tensor              # (T_up, 3) world frame
    base_ang_vel: torch.Tensor              # (T_up, 3) world frame

    actuated_idx: torch.LongTensor          # (n_actuated,) into the n_joints axis
    passive_idx: torch.LongTensor           # (n_joints - n_actuated,)
    actuated_names: list[str]               # MJCF actuator order

    @classmethod
    def load(
        cls,
        npz_path: Union[str, Path],
        asimov_xml: Union[str, Path],
        asimov_urdf: Union[str, Path],
        *,
        physics_fps: int = 60,
        src_fps: int = 30,
        device: Union[str, torch.device] = "cpu",
    ) -> "MotionRef":
        device = torch.device(device)
        if physics_fps % src_fps != 0:
            raise ValueError(
                f"physics_fps ({physics_fps}) must be a multiple of src_fps ({src_fps})"
            )
        upsample = physics_fps // src_fps

        data = np.load(npz_path)
        base_pos_src = torch.tensor(data["base_frame_pos"], dtype=torch.float32, device=device)
        base_quat_wxyz = torch.tensor(data["base_frame_wxyz"], dtype=torch.float32, device=device)
        joint_q_src = torch.tensor(data["joint_angles"], dtype=torch.float32, device=device)

        # wxyz → xyzw
        base_quat_src = torch.stack(
            [base_quat_wxyz[..., 1], base_quat_wxyz[..., 2],
             base_quat_wxyz[..., 3], base_quat_wxyz[..., 0]],
            dim=-1,
        )
        # Normalise (the optimisation output may drift slightly off unit length).
        base_quat_src = base_quat_src / torch.linalg.norm(
            base_quat_src, dim=-1, keepdim=True
        ).clamp(min=1e-9)

        T_src = base_pos_src.shape[0]
        n_joints = joint_q_src.shape[1]

        # ---- Resolve actuated joint mapping against URDF joint order.
        # The Asimov MJCF intentionally has no <actuator> block (it's added
        # programmatically downstream), so use the canonical list shipped
        # with the LAFAN1 retargeter. Cross-check against the MJCF when the
        # block IS present — useful guardrail if the upstream xml changes.
        mj_model = mujoco.MjModel.from_xml_path(str(asimov_xml))
        if mj_model.nu > 0:
            actuated_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT,
                                  int(mj_model.actuator_trnid[i, 0]))
                for i in range(mj_model.nu)
            ]
            if actuated_names != ASIMOV_ACTUATED_JOINT_NAMES:
                raise ValueError(
                    "MJCF actuator list disagrees with retargeting.joint_map. "
                    "Update one of them.\n"
                    f"  MJCF: {actuated_names}\n"
                    f"  ours: {ASIMOV_ACTUATED_JOINT_NAMES}"
                )
        else:
            actuated_names = list(ASIMOV_ACTUATED_JOINT_NAMES)

        urdf = yourdfpy.URDF.load(
            str(asimov_urdf), mesh_dir=str(Path(asimov_urdf).parent)
        )
        urdf_joint_order = list(urdf.actuated_joint_names)
        if len(urdf_joint_order) != n_joints:
            raise ValueError(
                f"URDF reports {len(urdf_joint_order)} actuated joints; "
                f"npz has {n_joints} columns in joint_angles."
            )

        actuated_idx = torch.tensor(
            [urdf_joint_order.index(n) for n in actuated_names],
            dtype=torch.long, device=device,
        )
        actuated_set = set(actuated_idx.tolist())
        passive_idx = torch.tensor(
            [i for i in range(n_joints) if i not in actuated_set],
            dtype=torch.long, device=device,
        )

        # ---- Build per-upsampled-frame interpolation lookups.
        T_up = (T_src - 1) * upsample + 1
        t_idx = torch.arange(T_up, device=device, dtype=torch.float32)
        t_src = t_idx / upsample                                   # in [0, T_src-1]
        # clamp(max=T_src - 2) ensures i_lo + 1 is always valid; for the very last
        # upsampled index t_src == T_src - 1 we get alpha == 1 and pull only from i_hi.
        i_lo = t_src.long().clamp(max=T_src - 2)
        alpha = (t_src - i_lo.float()).unsqueeze(-1)               # (T_up, 1)

        base_pos_up = (1.0 - alpha) * base_pos_src[i_lo] + alpha * base_pos_src[i_lo + 1]
        joint_q_up = (1.0 - alpha) * joint_q_src[i_lo] + alpha * joint_q_src[i_lo + 1]
        base_quat_up = slerp(
            base_quat_src[i_lo], base_quat_src[i_lo + 1], alpha.squeeze(-1)
        )

        # ---- Finite-difference velocities at physics dt.
        dt = 1.0 / physics_fps
        joint_qd_up = torch.zeros_like(joint_q_up)
        joint_qd_up[1:] = (joint_q_up[1:] - joint_q_up[:-1]) / dt
        joint_qd_up[0] = joint_qd_up[1]                            # extrapolate one step

        base_lin_vel = torch.zeros_like(base_pos_up)
        base_lin_vel[1:] = (base_pos_up[1:] - base_pos_up[:-1]) / dt
        base_lin_vel[0] = base_lin_vel[1]

        # Angular velocity (world frame): ω = (2/dt) * log(q_{t+1} * q_t^-1).xyz
        base_ang_vel = torch.zeros_like(base_pos_up)
        if T_up > 1:
            q_prev_inv = quat_conj(base_quat_up[:-1])
            q_delta = quat_mul(base_quat_up[1:], q_prev_inv)        # (T_up - 1, 4)
            base_ang_vel[1:] = (2.0 / dt) * quat_log_xyz(q_delta)
            base_ang_vel[0] = base_ang_vel[1]

        return cls(
            src_fps=src_fps, physics_fps=physics_fps, upsample=upsample,
            T_src=T_src, T_up=T_up, n_joints=n_joints,
            n_actuated=len(actuated_names), device=device,
            base_pos=base_pos_up, base_quat=base_quat_up,
            joint_q=joint_q_up, joint_qd=joint_qd_up,
            base_lin_vel=base_lin_vel, base_ang_vel=base_ang_vel,
            actuated_idx=actuated_idx, passive_idx=passive_idx,
            actuated_names=actuated_names,
        )

    # ----- Per-env access -----

    def sample_index(self, n_envs: int, max_lookahead: int) -> torch.LongTensor:
        """Uniform random integer index in [0, T_up - max_lookahead) per env."""
        hi = max(1, self.T_up - max_lookahead)
        return torch.randint(0, hi, (n_envs,), device=self.device)

    def at(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice all per-frame fields at the given indices (shape (B,))."""
        return {
            "base_pos":     self.base_pos[idx],
            "base_quat":    self.base_quat[idx],
            "joint_q":      self.joint_q[idx],
            "joint_qd":     self.joint_qd[idx],
            "base_lin_vel": self.base_lin_vel[idx],
            "base_ang_vel": self.base_ang_vel[idx],
        }

    def lookahead_actuated(
        self, idx: torch.Tensor, K: int, stride: int
    ) -> torch.Tensor:
        """Future actuated-joint targets. Returns (B, K, n_actuated)."""
        offsets = torch.arange(1, K + 1, device=idx.device) * stride
        future_idx = (idx.unsqueeze(-1) + offsets.unsqueeze(0)).clamp(max=self.T_up - 1)
        future_q = self.joint_q[future_idx]                        # (B, K, n_joints)
        return future_q[..., self.actuated_idx]
