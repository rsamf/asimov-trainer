"""Rollout a policy (or replay reference) and write a viser-compatible .npz.

Usage:
    # Phase 1: drive env with action=0 (residual-action design plays the reference)
    .venv/bin/python -m dancer.eval --replay-reference

    # Eval a trained checkpoint:
    .venv/bin/python -m dancer.eval --checkpoint runs/<run>/ckpt_final.pt

Outputs `eval.npz` with the same schema the existing viser viewer reads:
    base_frame_pos:  (T, 3)
    base_frame_wxyz: (T, 4) wxyz   ← viewer expects wxyz; we convert here
    joint_angles:    (T, 27)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from .env.dance_env import DanceEnv, EnvConfig
from .env.reward import RewardWeights
from .policy.networks import ActorCritic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-reference", action="store_true",
                   help="Drive env with action=0 (residual design plays the reference).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Load model state from this .pt (skip --replay-reference).")
    p.add_argument("--config", type=Path, default=Path("dancer/configs/train.yaml"),
                   help="Hydra config root to resolve from.")
    p.add_argument("--motion", type=Path, default=None,
                   help="Override the reference motion .npz path.")
    p.add_argument("--n-steps", type=int, default=600,
                   help="Number of policy steps to roll out.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=Path, default=Path("eval.npz"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_config(config_path: Path):
    """Resolve the train.yaml + its defaults list into one DictConfig.

    Uses Hydra's compose API so the same configs work both here and under
    `python -m dancer.train`.
    """
    from hydra import compose, initialize_config_dir
    cfg_dir = config_path.parent.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir),
                               job_name="dancer_eval"):
        return compose(config_name=config_path.stem)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        wp.set_device(args.device)

    cfg = load_config(args.config)

    env_cfg = EnvConfig(
        robot_xml=cfg.env.robot_xml,
        robot_urdf=cfg.env.robot_xml.replace("asimov.xml", "asimov.urdf"),
        motion_npz=str(args.motion) if args.motion else cfg.env.motion_npz,
        num_envs=1,                                 # single env for eval/playback
        dt=float(cfg.env.dt),
        control_decimation=int(cfg.env.control_decimation),
        kp=float(cfg.env.kp),
        kd=float(cfg.env.kd),
        action_scale=float(cfg.env.action_scale),
        episode_len=int(args.n_steps + 1),          # don't terminate early
        lookahead_K=int(cfg.env.lookahead_K),
        z_fall=-1e9,                                # disable fall-termination for eval
        up_dot_min=-1.0,
        joint_err_done=1e9,                         # disable joint-err termination
    )
    weights = RewardWeights(**OmegaConf.to_container(cfg.reward))

    env = DanceEnv(env_cfg, weights, device=args.device)

    model: Optional[ActorCritic] = None
    if args.checkpoint is not None:
        model = ActorCritic(
            obs_dim=env.obs_dim, act_dim=env.act_dim,
            hidden_sizes=tuple(cfg.network.hidden_sizes),
            activation=str(cfg.network.activation),
            log_std_init=float(cfg.network.log_std_init),
            learnable_log_std=bool(cfg.network.learnable_log_std),
        ).to(args.device)
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    elif not args.replay_reference:
        raise SystemExit("Pass either --replay-reference or --checkpoint")

    # Force start at phase 0 (the entire motion) by overriding the random reset.
    env.phase_index[:] = 0
    env.steps_in_episode[:] = 0
    env.last_action[:] = 0.0
    r0 = env.motion.at(env.phase_index)
    env.sim.reset_idx(
        env_ids=torch.zeros(1, dtype=torch.long, device=args.device),
        base_pos=r0["base_pos"],
        base_quat_xyzw=r0["base_quat"],
        base_lin_vel=r0["base_lin_vel"],
        base_ang_vel=r0["base_ang_vel"],
        hinge_q=r0["joint_q"],
        hinge_qd=r0["joint_qd"],
    )

    base_pos = np.zeros((args.n_steps, 3), dtype=np.float32)
    base_wxyz = np.zeros((args.n_steps, 4), dtype=np.float32)
    joint_angles = np.zeros((args.n_steps, env.sim.n_hinges), dtype=np.float32)

    obs = env._compute_obs()
    for t in range(args.n_steps):
        if args.replay_reference:
            action = torch.zeros(1, env.act_dim, device=args.device)
        else:
            with torch.no_grad():
                action, _, _ = model.act(obs, greedy=True)
        obs, _, _, _ = env.step(action)
        bp = env.sim.base_pos[0].detach().cpu().numpy()
        bq = env.sim.base_quat[0].detach().cpu().numpy()  # xyzw
        jq = env.sim.joint_q[0].detach().cpu().numpy()
        base_pos[t] = bp
        # viewer expects wxyz
        base_wxyz[t] = np.array([bq[3], bq[0], bq[1], bq[2]], dtype=np.float32)
        joint_angles[t] = jq

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             base_frame_pos=base_pos,
             base_frame_wxyz=base_wxyz,
             joint_angles=joint_angles)
    print(f"wrote {args.out}  ({args.n_steps} steps)")
    print(f"  base_pos z range: [{base_pos[:,2].min():.3f}, {base_pos[:,2].max():.3f}]")
    print(f"  joint range: [{joint_angles.min():.3f}, {joint_angles.max():.3f}]")
    print(f"\nReplay with:  .venv/bin/python -m retargeting.pyroki.viewer {args.out}")


if __name__ == "__main__":
    main()
