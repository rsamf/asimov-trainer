"""Vectorized DeepMimic-style env for Asimov dance imitation.

Glues `NewtonSim` + `MotionRef` + reward into a gym-ish loop with batched
torch tensors throughout. Residual-action design: action=0 plays the
reference (so a freshly initialised policy starts on-policy).

Observation (matches the plan + memory):
    - Own state (87 dims):  base_lin_vel(3) + base_ang_vel(3) + base_quat(4)
                            + joint_q(27) + joint_qd(27) + last_action(23)
    - Lookahead (K*23):     future actuated-joint targets at next K policy
                            steps (each = control_decimation physics frames).
    NO phase scalar, NO gravity projection.

Reset = Reference State Initialisation. We pick a random integer phase index
into the pre-upsampled motion, copy that frame's full state into the sim, and
zero the per-env episode counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .motion import MotionRef
from .reward import RewardWeights, compute_reward, is_fallen
from .sim import NewtonSim


@dataclass
class EnvConfig:
    robot_xml: str
    robot_urdf: str
    motion_npz: str
    num_envs: int = 1024
    dt: float = 1.0 / 60.0          # physics dt
    control_decimation: int = 2     # policy steps per `control_decimation` physics steps
    kp: float = 100.0
    kd: float = 5.0
    action_scale: float = 0.5
    episode_len: int = 600          # max policy steps per episode
    lookahead_K: int = 4
    z_fall: float = 0.30
    up_dot_min: float = 0.5
    joint_err_done: float = 1.5     # mean |q_act - ref_q_act| above this → done


class DanceEnv:
    """Vectorized env. All tensors on `device` (typically cuda:0)."""

    def __init__(
        self,
        cfg: EnvConfig,
        reward_weights: RewardWeights,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self.cfg = cfg
        self.reward_weights = reward_weights
        self.device = torch.device(device)
        self.num_envs = cfg.num_envs

        # Motion lives at physics rate so phase_index increments by an integer
        # per physics step (or by control_decimation per policy step).
        physics_fps = int(round(1.0 / cfg.dt))
        self.motion = MotionRef.load(
            cfg.motion_npz, cfg.robot_xml, cfg.robot_urdf,
            physics_fps=physics_fps, src_fps=30, device=self.device,
        )
        self.sim = NewtonSim(
            cfg.robot_xml,
            actuated_joint_names=self.motion.actuated_names,
            num_envs=cfg.num_envs,
            dt=cfg.dt,
            kp=cfg.kp,
            kd=cfg.kd,
            control_decimation=cfg.control_decimation,
            device=device,
        )

        # Per-env state.
        self.phase_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps_in_episode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(self.num_envs, self.sim.n_actuated,
                                       dtype=torch.float32, device=self.device)
        # Buffer indices used by reset to sample new phases.
        self._max_lookahead_phys = cfg.lookahead_K * cfg.control_decimation

        # Dimensions.
        self.act_dim: int = self.sim.n_actuated                                # 23
        self.obs_dim: int = (3 + 3 + 4
                             + self.sim.n_hinges + self.sim.n_hinges
                             + self.act_dim
                             + cfg.lookahead_K * self.act_dim)

        # First reset on all envs.
        self.reset_all()

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------

    def reset_all(self) -> torch.Tensor:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._compute_obs()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        new_phases = self.motion.sample_index(env_ids.numel(), self._max_lookahead_phys)
        self.phase_index[env_ids] = new_phases
        r = self.motion.at(new_phases)
        self.sim.reset_idx(
            env_ids=env_ids,
            base_pos=r["base_pos"],
            base_quat_xyzw=r["base_quat"],
            base_lin_vel=r["base_lin_vel"],
            base_ang_vel=r["base_ang_vel"],
            hinge_q=r["joint_q"],
            hinge_qd=r["joint_qd"],
        )
        self.steps_in_episode[env_ids] = 0
        self.last_action[env_ids] = 0.0

    # ----------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """action: (N, n_actuated). Returns (obs, reward, done, info).

        On done, env_ids are auto-reset and the returned obs reflects the
        NEW episode's first frame (standard gym convention). Pre-reset
        terminal obs is stashed in info["terminal_obs"] for PPO bootstrapping.
        """
        # 1. Reference state at the env's current phase (pre-step).
        ref = self.motion.at(self.phase_index)

        # 2. PD target on the 23 actuated joints = ref_q[actuated] + residual.
        action = action.to(self.device, dtype=torch.float32)
        ref_target = ref["joint_q"][:, self.motion.actuated_idx]
        target = ref_target + torch.tanh(action) * self.cfg.action_scale

        # 3. Physics step.
        self.sim.step(target)

        # 4. Advance phase by control_decimation; clamp at motion end.
        next_phase = self.phase_index + self.cfg.control_decimation
        motion_end = next_phase >= (self.motion.T_up - 1)
        self.phase_index = next_phase.clamp(max=self.motion.T_up - 1)
        self.steps_in_episode += 1
        self.last_action = action

        # 5. Reward — uses POST-step robot state vs the phase-aligned ref
        # that the action was supposed to drive us toward. We re-fetch ref
        # at the (advanced) phase so the comparison is consistent.
        ref_post = self.motion.at(self.phase_index)
        fallen = is_fallen(
            self.sim.base_pos, self.sim.base_quat,
            z_fall=self.cfg.z_fall, up_dot_min=self.cfg.up_dot_min,
        )
        reward, rinfo = compute_reward(
            joint_q=self.sim.joint_q, joint_qd=self.sim.joint_qd,
            base_pos=self.sim.base_pos, base_quat_xyzw=self.sim.base_quat,
            ref_joint_q=ref_post["joint_q"], ref_joint_qd=ref_post["joint_qd"],
            ref_base_pos=ref_post["base_pos"], ref_base_quat_xyzw=ref_post["base_quat"],
            action=action, actuated_idx=self.motion.actuated_idx,
            weights=self.reward_weights, fallen=fallen,
        )

        # 6. Done flags.
        joint_err = (
            (self.sim.joint_q[:, self.motion.actuated_idx]
             - ref_post["joint_q"][:, self.motion.actuated_idx])
            .abs().mean(dim=-1)
        )
        joint_blowup = joint_err > self.cfg.joint_err_done
        timeout = self.steps_in_episode >= self.cfg.episode_len
        done = fallen | joint_blowup | motion_end | timeout

        info = dict(rinfo)
        info["motion_end"] = motion_end
        info["timeout"] = timeout
        info["joint_blowup"] = joint_blowup
        info["joint_err"] = joint_err

        # 7. Auto-reset, gym-style.
        obs_terminal = self._compute_obs()
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)
        info["terminal_obs"] = obs_terminal
        obs = self._compute_obs()
        return obs, reward, done, info

    # ----------------------------------------------------------------------
    # Observation
    # ----------------------------------------------------------------------

    def _compute_obs(self) -> torch.Tensor:
        own = torch.cat([
            self.sim.base_lin_vel,                  # (N, 3)
            self.sim.base_ang_vel,                  # (N, 3)
            self.sim.base_quat,                     # (N, 4) xyzw
            self.sim.joint_q,                       # (N, 27)
            self.sim.joint_qd,                      # (N, 27)
            self.last_action,                       # (N, 23)
        ], dim=-1)
        lookahead = self.motion.lookahead_actuated(
            self.phase_index,
            K=self.cfg.lookahead_K,
            stride=self.cfg.control_decimation,
        ).reshape(self.num_envs, -1)
        return torch.cat([own, lookahead], dim=-1)
