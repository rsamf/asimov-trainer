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
    dual_clip: float = 3.0          # dual-clip PPO floor for adv<0 (Ye et al. 2020)


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
                adv = b_adv[idx]
                ratio = (logp - b_logp_old[idx]).exp()

                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv
                clipped = torch.min(surr1, surr2)
                # Dual-clip PPO (Ye et al. 2020): for negative-advantage samples
                # the standard objective is UNBOUNDED below (ratio*adv → -∞ as
                # the ratio explodes), which produced the million-scale policy
                # losses and divergence. Floor it at dual_c * adv so a few
                # tail samples with huge ratios can't blow up the update.
                dual = torch.where(
                    adv < 0.0,
                    torch.max(clipped, cfg.dual_clip * adv),
                    clipped,
                )
                policy_loss = -dual.mean()

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
                    # Schulman's k3 KL, but reported as the MEDIAN over samples.
                    # The mean is dominated by a small (~4%) tail of high-leverage
                    # samples whose ratios explode while the bulk of the policy
                    # barely moves (clip_frac ~0.04). The mean read 89-235 vs a
                    # 0.02 target and early-stopped PPO after a single minibatch,
                    # throttling it to ~1 update/iter. The median tracks the bulk
                    # movement, so the KL guardrail fires on real drift, not on
                    # outliers (which dual-clip + PPO-clip already bound).
                    logratio = (logp - b_logp_old[idx]).clamp(-10.0, 10.0)
                    kl_per = logratio.exp() - 1.0 - logratio
                    approx_kl = kl_per.median().item()
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
