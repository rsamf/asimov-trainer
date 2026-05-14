"""PPO clipped-surrogate update."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..policy.networks import ActorCritic
from .rollout import RolloutData


@dataclass
class PPOConfig:
    n_epochs: int = 5
    mb_size: int = 16384
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3.0e-4
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.02


class PPOTrainer:
    def __init__(self, model: ActorCritic, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def update(self, data: RolloutData) -> dict[str, float]:
        cfg = self.cfg
        T, N = data.rewards.shape
        # Flatten time × env dims into one batch.
        b_obs = data.obs.reshape(T * N, -1)
        b_act = data.actions.reshape(T * N, -1)
        b_logp_old = data.logp.reshape(T * N)
        b_adv = data.advantages.reshape(T * N)
        b_val_old = data.values.reshape(T * N)
        b_ret = data.returns.reshape(T * N)

        # Advantage normalisation.
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        mb = min(cfg.mb_size, n)
        stats = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
        }
        n_updates = 0
        early_stop = False
        for epoch in range(cfg.n_epochs):
            perm = torch.randperm(n, device=b_obs.device)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                logp, value, entropy = self.model.evaluate(b_obs[idx], b_act[idx])
                ratio = (logp - b_logp_old[idx]).exp()

                surr1 = ratio * b_adv[idx]
                surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * b_adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = b_val_old[idx] + (value - b_val_old[idx]).clamp(
                    -cfg.clip, cfg.clip)
                value_loss = 0.5 * torch.max(
                    (value - b_ret[idx]) ** 2,
                    (value_clipped - b_ret[idx]) ** 2,
                ).mean()

                entropy_mean = entropy.mean()
                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_mean)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm).item()
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (b_logp_old[idx] - logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean().item()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["grad_norm"] += grad_norm
                n_updates += 1

                if approx_kl > 1.5 * cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        for k in stats:
            stats[k] /= max(1, n_updates)
        stats["lr"] = self.opt.param_groups[0]["lr"]
        stats["n_updates"] = n_updates
        return stats
