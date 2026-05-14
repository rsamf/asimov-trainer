"""PPO training entry point for Asimov dance imitation.

Hydra config under `dancer/configs/`. Logging via nebo (also writes scalar
metrics that show up under the run). Checkpoints + eval rollouts under
`runs/<run_name>/`.

Run:
    .venv/bin/python -m dancer.train
    .venv/bin/python -m dancer.train experiment_name=dance1_subject3_v2 algo.lr=1e-4
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import hydr8
import nebo as nb
import newton
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf

from .algos.ppo import PPOConfig, PPOTrainer
from .algos.rollout import RolloutBuffer
from .env.dance_env import DanceEnv, EnvConfig
from .env.reward import RewardWeights
from .policy.networks import ActorCritic


def _build_viewer(cfg: DictConfig, run_dir: Path):
    """Return a Newton viewer (or None) per cfg.viewer.kind."""
    kind = cfg.viewer.kind
    if kind is None or kind == "" or str(kind).lower() == "null":
        return None
    kind = str(kind).lower()
    if kind == "rerun":
        rrd = cfg.viewer.get("record_to_rrd", None)
        return newton.viewer.ViewerRerun(
            app_id="asimov-dance-train",
            record_to_rrd=str(rrd) if rrd else None,
        )
    if kind == "viser":
        return newton.viewer.ViewerViser()
    if kind == "gl":
        return newton.viewer.ViewerGL()
    if kind == "usd":
        return newton.viewer.ViewerUSD(output_path=str(run_dir / "viewer.usd"))
    raise ValueError(f"unknown viewer.kind: {kind!r}")


def _build_env(cfg: DictConfig, device: str) -> DanceEnv:
    env_cfg = EnvConfig(
        robot_xml=cfg.env.robot_xml,
        robot_urdf=cfg.env.robot_xml.replace("asimov.xml", "asimov.urdf"),
        motion_npz=cfg.env.motion_npz,
        num_envs=int(cfg.env.num_envs),
        dt=float(cfg.env.dt),
        control_decimation=int(cfg.env.control_decimation),
        kp=float(cfg.env.kp),
        kd=float(cfg.env.kd),
        action_scale=float(cfg.env.action_scale),
        episode_len=int(cfg.env.episode_len),
        lookahead_K=int(cfg.env.lookahead_K),
        z_fall=float(cfg.env.z_fall),
        up_dot_min=float(cfg.env.up_dot_min),
        joint_err_done=float(cfg.env.joint_err_done),
    )
    weights = RewardWeights(
        w_jp=float(cfg.reward.w_jp),
        w_jv=float(cfg.reward.w_jv),
        w_rh=float(cfg.reward.w_rh),
        w_rp=float(cfg.reward.w_rp),
        w_rq=float(cfg.reward.w_rq),
        w_fall=float(cfg.reward.w_fall),
        w_action=float(cfg.reward.w_action),
        w_alive=float(cfg.reward.w_alive),
    )
    return DanceEnv(env_cfg, weights, device=device)


def _build_policy(cfg: DictConfig, obs_dim: int, act_dim: int, device: str) -> ActorCritic:
    return ActorCritic(
        obs_dim=obs_dim, act_dim=act_dim,
        hidden_sizes=tuple(cfg.network.hidden_sizes),
        activation=str(cfg.network.activation),
        log_std_init=float(cfg.network.log_std_init),
        learnable_log_std=bool(cfg.network.learnable_log_std),
    ).to(device)


def _ppo_config_from(cfg: DictConfig) -> PPOConfig:
    return PPOConfig(
        n_epochs=int(cfg.algo.n_epochs),
        mb_size=int(cfg.algo.mb_size),
        clip=float(cfg.algo.clip),
        gamma=float(cfg.algo.gamma),
        lam=float(cfg.algo.lam),
        lr=float(cfg.algo.lr),
        entropy_coef=float(cfg.algo.entropy_coef),
        value_coef=float(cfg.algo.value_coef),
        max_grad_norm=float(cfg.algo.max_grad_norm),
        target_kl=float(cfg.algo.target_kl),
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    hydr8.init(cfg)
    device = str(cfg.device)
    if device.startswith("cuda"):
        wp.set_device(device)
    torch.manual_seed(int(cfg.seed))

    run_name = f"{cfg.experiment_name}-{int(time.time())}"
    run_dir = Path(cfg.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {run_name} ==  device={device}  log_dir={run_dir}")
    nb.init(mode="auto")

    with nb.start_run(name=run_name,
                      config=OmegaConf.to_container(cfg, resolve=True)) as run:
        env = _build_env(cfg, device=device)
        print(f"env: num_envs={env.num_envs}  obs_dim={env.obs_dim}  act_dim={env.act_dim}")

        viewer = _build_viewer(cfg, run_dir)
        if viewer is not None:
            viewer.set_model(env.sim.model)
            viewer.set_world_offsets((
                float(cfg.viewer.world_offset_x),
                float(cfg.viewer.world_offset_y),
                0.0,
            ))
            print(f"viewer: {cfg.viewer.kind} (every {cfg.viewer.every_n_steps} steps)")
        viewer_every = int(cfg.viewer.every_n_steps)

        model = _build_policy(cfg, env.obs_dim, env.act_dim, device)
        ppo = PPOTrainer(model, _ppo_config_from(cfg))
        buf = RolloutBuffer(env.num_envs, int(cfg.algo.rollout_len),
                            env.obs_dim, env.act_dim, torch.device(device))

        obs = env.reset_all()
        ep_returns = torch.zeros(env.num_envs, device=device)
        ep_lens = torch.zeros(env.num_envs, device=device)
        global_step = 0
        sim_time = 0.0
        sim_dt = float(cfg.env.dt) * int(cfg.env.control_decimation)

        for it in range(int(cfg.algo.iterations)):
            buf.reset()
            t0 = time.perf_counter()
            finished_returns: list[float] = []
            finished_lens: list[int] = []
            n_falls = 0

            for rt in range(buf.rollout_len):
                action, logp, value = model.act(obs)
                next_obs, reward, done, info = env.step(action)
                buf.add(obs, action, logp, value, reward, done)
                obs = next_obs
                ep_returns += reward
                ep_lens += 1.0
                # Track episode summaries.
                done_idx = done.nonzero(as_tuple=False).flatten()
                if done_idx.numel() > 0:
                    finished_returns.extend(ep_returns[done_idx].tolist())
                    finished_lens.extend(ep_lens[done_idx].tolist())
                    n_falls += int(info["fallen"][done_idx].sum().item())
                    ep_returns[done_idx] = 0.0
                    ep_lens[done_idx] = 0.0
                global_step += env.num_envs
                sim_time += sim_dt
                # Optional viewer push.
                if viewer is not None and (rt % viewer_every == 0):
                    viewer.begin_frame(sim_time)
                    viewer.log_state(env.sim.state_0)
                    viewer.end_frame()

            with torch.no_grad():
                _, _, last_value = model.act(obs)
            data = buf.compute_gae(last_value, ppo.cfg.gamma, ppo.cfg.lam)
            stats = ppo.update(data)
            collect_t = time.perf_counter() - t0

            avg_ret = (sum(finished_returns) / max(1, len(finished_returns))
                       if finished_returns else float(ep_returns.mean()))
            avg_len = (sum(finished_lens) / max(1, len(finished_lens))
                       if finished_lens else float(ep_lens.mean()))

            print(f"it {it:4d}  step {global_step:>9d}  "
                  f"R̄={avg_ret:6.2f}  ℓ̄={avg_len:5.1f}  "
                  f"fall%={n_falls/max(1,len(finished_returns)):.0%}  "
                  f"pi_l={stats['policy_loss']:+.3f}  "
                  f"v_l={stats['value_loss']:.3f}  "
                  f"H={stats['entropy']:.2f}  "
                  f"KL={stats['kl']:.3f}  "
                  f"t={collect_t:.1f}s")

            nb.log_metric("train/avg_return", float(avg_ret), step=it)
            nb.log_metric("train/avg_episode_len", float(avg_len), step=it)
            nb.log_metric("train/fall_fraction",
                          float(n_falls / max(1, len(finished_returns))), step=it)
            nb.log_metric("train/n_episodes", float(len(finished_returns)), step=it)
            for k, v in stats.items():
                nb.log_metric(f"train/{k}", float(v), step=it)
            nb.log_metric("train/sec_per_iter", collect_t, step=it)

            if it % int(cfg.eval.every) == 0:
                _save_checkpoint(run_dir, it, model, ppo)

        _save_checkpoint(run_dir, "final", model, ppo)
        if viewer is not None:
            viewer.close()
        print(f"done. checkpoints in {run_dir}")


def _save_checkpoint(run_dir: Path, tag, model: ActorCritic, ppo: PPOTrainer) -> None:
    path = run_dir / f"ckpt_{tag}.pt"
    torch.save({
        "model": model.state_dict(),
        "opt": ppo.opt.state_dict(),
    }, path)
    print(f"  ↳ checkpoint: {path}")


if __name__ == "__main__":
    main()
