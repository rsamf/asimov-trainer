"""Newton physics wrapper for vectorized Asimov-v1 imitation training.

Loads the Asimov MJCF, configures per-joint PD (ke/kd for the 23 actuated
joints, zero for the 4 passive ones so they evolve under their MJCF spring),
replicates the robot across `num_envs` worlds, and exposes a small torch-
tensor surface for stepping and state access.

Newton stores all per-world state in flat arrays (`state.joint_q`,
`state.joint_qd`, `state.body_q`, ...). We map between flat and per-env
tensors by slicing fixed-size blocks; the layout is:

    joint_q[e * nq_per_env : (e + 1) * nq_per_env]
        = [freejoint_qpos(7), hinge_q[0..n_joints-1]]
    joint_qd[e * nqd_per_env : (e + 1) * nqd_per_env]
        = [freejoint_qvel(6), hinge_qd[0..n_joints-1]]

The freejoint qpos in Newton is `(x, y, z, qx, qy, qz, qw)` (Warp's `quat`
type is xyzw) — matches this project's convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import mujoco as _mj
import newton
import numpy as _np
import torch
import warp as wp


# ---------------------------------------------------------------------------
# mujoco 3.8.0 / mujoco-warp / Newton compatibility shim.
#
# Newton's SolverMuJoCo converts a Newton Model into mujoco's MjSpec by calling
# `MjsBody.add_joint(stiffness=..., ref=..., damping=..., ...)`. The values
# come from Warp arrays (float32). In mujoco 3.8 the pybind11 binding for
# these kwargs accepts Python `float` and `np.float64` but rejects
# `np.float32` with `TypeError: stiffness should be a numeric scalar or list`.
#
# Wrap MjsBody.add_joint once globally to coerce float32 scalars to Python
# floats before they hit the binding. Idempotent.
# ---------------------------------------------------------------------------

def _install_mjs_add_joint_patch() -> None:
    spec = _mj.MjSpec()
    body_cls = type(spec.worldbody.add_body())
    orig = body_cls.add_joint
    if getattr(orig, "_dancer_patched", False):
        return

    def _coerce(v):
        if isinstance(v, _np.floating):
            return float(v)
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.bool_):
            return bool(v)
        return v

    def add_joint(self, *args, **kwargs):
        kwargs = {k: _coerce(v) for k, v in kwargs.items()}
        return orig(self, *args, **kwargs)
    add_joint._dancer_patched = True
    body_cls.add_joint = add_joint


_install_mjs_add_joint_patch()


def _joint_short_name(label: str) -> str:
    """Strip the URDF/MJCF path prefix from a Newton joint label."""
    return label.split("/")[-1]


class NewtonSim:
    """Vectorized Asimov-v1 simulator. One model, `num_envs` worlds."""

    def __init__(
        self,
        asimov_xml: Union[str, Path],
        *,
        actuated_joint_names: list[str],
        num_envs: int,
        dt: float,
        kp: float,
        kd: float,
        control_decimation: int = 1,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.kp = float(kp)
        self.kd = float(kd)
        self.control_decimation = int(control_decimation)
        self.device = torch.device(device)
        self._wp_device = "cuda:0" if self.device.type == "cuda" else "cpu"
        # Newton/Warp picks the default Warp device when building the model.
        # Pin it to match our target so model arrays and torch live on the
        # same device (necessary for wp.to_torch zero-copy).
        wp.set_device(self._wp_device)

        # ---- Single-robot template ----------------------------------------
        single = newton.ModelBuilder()
        single.add_mjcf(
            str(asimov_xml),
            floating=None,
            enable_self_collisions=False,    # MJCF's <contact><exclude> blocks
                                             # aren't reliably preserved through
                                             # Newton's converter; disable global
                                             # self-collision to keep step stable.
        )
        self._joint_names: list[str] = [_joint_short_name(l) for l in single.joint_label]
        self.n_joints_per_env: int = single.joint_count           # incl. freejoint
        self.nq_per_env: int = single.joint_coord_count           # per env (e.g. 34 for Asimov)
        self.nqd_per_env: int = single.joint_dof_count            # per env (e.g. 33 for Asimov)

        # Resolve actuated-joint DoF indices in the per-env DoF block.
        # joint_qd_start[i] is the DoF offset (within an env) where joint i's
        # velocities start; for a 1-DoF hinge, that single index is the joint's
        # DoF index.
        joint_qd_start = list(single.joint_qd_start)              # length = n_joints_per_env
        actuated_dof_local: list[int] = []
        name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
        for n in actuated_joint_names:
            if n not in name_to_idx:
                raise KeyError(f"actuated joint {n!r} not found in MJCF "
                               f"(joint names: {self._joint_names[:5]}...)")
            j = name_to_idx[n]
            actuated_dof_local.append(joint_qd_start[j])
        self.n_actuated: int = len(actuated_joint_names)
        self.actuated_joint_names: list[str] = list(actuated_joint_names)
        self.actuated_dof_local = torch.tensor(
            actuated_dof_local, dtype=torch.long, device=self.device
        )

        # ---- Configure PD per joint --------------------------------------
        # By default, zero PD on every DoF. Then set kp/kd/mode for the
        # actuated DoFs only. Passive joints evolve under whatever
        # joint_armature/damping the MJCF loader gave them.
        for d in range(single.joint_dof_count):
            single.joint_target_ke[d] = 0.0
            single.joint_target_kd[d] = 0.0
            single.joint_target_mode[d] = int(newton.JointTargetMode.NONE)
        for d in actuated_dof_local:
            single.joint_target_ke[d] = self.kp
            single.joint_target_kd[d] = self.kd
            single.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)

        # ---- Replicate, ground plane, finalize ----------------------------
        builder = newton.ModelBuilder()
        # Contact stiffness/damping/friction defaults that match the H1
        # humanoid example — Newton's out-of-box ke is too high for the
        # Asimov's thin foot capsules and the contact solver explodes within
        # a few steps of ground touchdown.
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.replicate(single, self.num_envs)
        builder.add_ground_plane()
        self.model = builder.finalize()

        # ---- Solver + state ----------------------------------------------
        # SolverMuJoCo is the most accurate articulated-robot backend but
        # requires the `mujoco_warp` extension which isn't always available.
        # Fall back to SolverFeatherstone (Newton's built-in articulated-body
        # solver — pure Warp, no external deps).
        try:
            # iterations/ls_iterations from the Newton H1 example; raises
            # contact-solver fidelity considerably vs the defaults and keeps
            # the humanoid from punching through the floor at reset.
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=100,
                ls_iterations=50,
                njmax=200,
                nconmax=400,
            )
        except (ImportError, RuntimeError):
            self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Initialise state from the model's joint_q / joint_qd (rest pose).
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- Pre-compute index buffers for fast per-env writes ----------
        # Global DoF index for env e, actuated slot k:
        #     g = e * nqd_per_env + actuated_dof_local[k]
        # We'll materialise that lookup table once.
        env_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.nqd_per_env
        )
        self.actuated_dof_global = (
            env_offsets.unsqueeze(1) + self.actuated_dof_local.unsqueeze(0)
        )  # (num_envs, n_actuated)

        # Same for qpos hinge slice (skip the 7-element freejoint per env).
        joint_q_start = list(single.joint_q_start)
        hinge_q_local: list[int] = []
        hinge_qd_local: list[int] = []
        self.hinge_joint_indices: list[int] = []
        for i, jt in enumerate(single.joint_type):
            if jt == newton.JointType.FREE:
                continue
            hinge_q_local.append(joint_q_start[i])
            hinge_qd_local.append(joint_qd_start[i])
            self.hinge_joint_indices.append(i)
        self.n_hinges: int = len(hinge_q_local)
        self.hinge_q_local = torch.tensor(hinge_q_local, dtype=torch.long, device=self.device)
        self.hinge_qd_local = torch.tensor(hinge_qd_local, dtype=torch.long, device=self.device)
        env_q_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.nq_per_env
        )
        env_qd_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.nqd_per_env
        )
        self.hinge_q_global = env_q_offsets.unsqueeze(1) + self.hinge_q_local.unsqueeze(0)
        self.hinge_qd_global = env_qd_offsets.unsqueeze(1) + self.hinge_qd_local.unsqueeze(0)

        # Freejoint slice indices.
        self.freejoint_q_global = env_q_offsets.unsqueeze(1) + torch.arange(
            7, device=self.device, dtype=torch.long
        ).unsqueeze(0)  # (num_envs, 7)
        self.freejoint_qd_global = env_qd_offsets.unsqueeze(1) + torch.arange(
            6, device=self.device, dtype=torch.long
        ).unsqueeze(0)  # (num_envs, 6)

        # Pelvis body index in the global body_q array. Body 0 in a single
        # template is the world body (ground), and the next body is the
        # pelvis_link (the freejoint child).
        body_offsets = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        ) * (self.model.body_count // self.num_envs)
        # World body in the replicated model is shared per env or global? Newton
        # replicate creates one ground-plane world body and N robots' bodies.
        # We resolve "pelvis_link" body indices below empirically.
        self._pelvis_global_idx = self._resolve_pelvis_indices()

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _resolve_pelvis_indices(self) -> torch.LongTensor:
        """Find each env's pelvis_link body id in the finalized model."""
        n_body = int(self.model.body_count)
        indices: list[int] = []
        # The model exposes body keys as a list of strings; the pelvis label
        # ends in '/pelvis_link'. Each replicated env's body has the same suffix.
        body_keys = list(self.model.body_key) if hasattr(self.model, "body_key") else []
        if body_keys:
            for i, k in enumerate(body_keys):
                if k.endswith("/pelvis_link") or k.endswith("pelvis_link"):
                    indices.append(i)
        # If the body-key approach didn't yield exactly num_envs hits, fall
        # back to assuming the first body of each env's robot is the pelvis
        # (Newton lays them out contiguously after replicate).
        if len(indices) != self.num_envs:
            bodies_per_env = n_body // self.num_envs
            # body 0 is typically the ground; pelvis follows.
            indices = [e * bodies_per_env + (1 if bodies_per_env > 1 else 0)
                       for e in range(self.num_envs)]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    # ----------------------------------------------------------------------
    # Zero-copy torch views into Newton state
    # ----------------------------------------------------------------------

    def _torch(self, arr: wp.array) -> torch.Tensor:
        return wp.to_torch(arr)

    @property
    def _joint_q_flat(self) -> torch.Tensor:
        return self._torch(self.state_0.joint_q)

    @property
    def _joint_qd_flat(self) -> torch.Tensor:
        return self._torch(self.state_0.joint_qd)

    @property
    def _body_q_flat(self) -> torch.Tensor:
        return self._torch(self.state_0.body_q)

    @property
    def _body_qd_flat(self) -> torch.Tensor:
        return self._torch(self.state_0.body_qd)

    @property
    def _ctrl_target_flat(self) -> torch.Tensor:
        return self._torch(self.control.joint_target_pos)

    # ----- Public per-env views (read-only convenience) -----

    @property
    def joint_q(self) -> torch.Tensor:
        """Hinge joint angles per env: (num_envs, n_hinges)."""
        return self._joint_q_flat[self.hinge_q_global]

    @property
    def joint_qd(self) -> torch.Tensor:
        """Hinge joint velocities per env: (num_envs, n_hinges)."""
        return self._joint_qd_flat[self.hinge_qd_global]

    @property
    def base_pos(self) -> torch.Tensor:
        """Pelvis world position from freejoint qpos: (num_envs, 3)."""
        return self._joint_q_flat[self.freejoint_q_global][:, :3]

    @property
    def base_quat(self) -> torch.Tensor:
        """Pelvis world quaternion xyzw from freejoint qpos: (num_envs, 4)."""
        return self._joint_q_flat[self.freejoint_q_global][:, 3:7]

    @property
    def base_lin_vel(self) -> torch.Tensor:
        """Pelvis linear velocity in world frame: (num_envs, 3)."""
        # Newton freejoint qd layout: (ang_vel(3), lin_vel(3)). Confirm at runtime.
        return self._joint_qd_flat[self.freejoint_qd_global][:, 3:6]

    @property
    def base_ang_vel(self) -> torch.Tensor:
        """Pelvis angular velocity in world frame: (num_envs, 3)."""
        return self._joint_qd_flat[self.freejoint_qd_global][:, 0:3]

    # ----------------------------------------------------------------------
    # Step / reset
    # ----------------------------------------------------------------------

    def step(self, actuated_target_pos: torch.Tensor) -> None:
        """Apply PD targets to actuated joints and advance physics.

        actuated_target_pos: (num_envs, n_actuated) torch tensor on self.device.
        """
        if actuated_target_pos.shape != (self.num_envs, self.n_actuated):
            raise ValueError(
                f"actuated_target_pos shape {actuated_target_pos.shape} "
                f"!= ({self.num_envs}, {self.n_actuated})"
            )
        # Scatter into the flat control buffer.
        targets_flat = self._ctrl_target_flat
        targets_flat[self.actuated_dof_global] = actuated_target_pos.to(targets_flat.dtype)

        # collide + substep loop. Substeps = control_decimation.
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.control_decimation):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        base_pos: torch.Tensor,
        base_quat_xyzw: torch.Tensor,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        hinge_q: torch.Tensor,
        hinge_qd: torch.Tensor,
    ) -> None:
        """Overwrite per-env state for the given envs.

        Shapes (M envs to reset):
          env_ids:        (M,) long
          base_pos:       (M, 3)
          base_quat_xyzw: (M, 4)
          base_lin_vel:   (M, 3)
          base_ang_vel:   (M, 3)
          hinge_q:        (M, n_hinges)
          hinge_qd:       (M, n_hinges)
        """
        env_ids = env_ids.to(self.device, dtype=torch.long)
        jq = self._joint_q_flat
        jqd = self._joint_qd_flat

        # Freejoint qpos: (x, y, z, qx, qy, qz, qw)
        fq_idx = self.freejoint_q_global[env_ids]                  # (M, 7)
        jq[fq_idx[:, 0:3]] = base_pos.to(jq.dtype)
        jq[fq_idx[:, 3:7]] = base_quat_xyzw.to(jq.dtype)

        # Freejoint qd: (ang(3), lin(3))
        fqd_idx = self.freejoint_qd_global[env_ids]                # (M, 6)
        jqd[fqd_idx[:, 0:3]] = base_ang_vel.to(jqd.dtype)
        jqd[fqd_idx[:, 3:6]] = base_lin_vel.to(jqd.dtype)

        # Hinge qpos/qd
        hq_idx = self.hinge_q_global[env_ids]                      # (M, n_hinges)
        hqd_idx = self.hinge_qd_global[env_ids]
        jq[hq_idx] = hinge_q.to(jq.dtype)
        jqd[hqd_idx] = hinge_qd.to(jqd.dtype)

        # Re-evaluate forward kinematics so body_q / body_qd reflect the new
        # joint state (otherwise the next step starts from stale body poses).
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)


def build_sim_from_config(
    asimov_xml: Union[str, Path],
    actuated_joint_names: list[str],
    *,
    num_envs: int,
    dt: float,
    kp: float,
    kd: float,
    control_decimation: int,
    device: str,
) -> NewtonSim:
    """Tiny factory used by dance_env.py + hydra wiring."""
    return NewtonSim(
        asimov_xml,
        actuated_joint_names=actuated_joint_names,
        num_envs=num_envs,
        dt=dt,
        kp=kp,
        kd=kd,
        control_decimation=control_decimation,
        device=device,
    )
