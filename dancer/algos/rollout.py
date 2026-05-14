"""Vectorised PPO rollout buffer with GAE advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutData:
    obs: torch.Tensor          # (T, N, obs_dim)
    actions: torch.Tensor      # (T, N, act_dim)
    logp: torch.Tensor         # (T, N)
    values: torch.Tensor       # (T, N)
    rewards: torch.Tensor      # (T, N)
    dones: torch.Tensor        # (T, N) float (1 if episode ended that step)
    advantages: torch.Tensor   # (T, N)
    returns: torch.Tensor      # (T, N)


class RolloutBuffer:
    """Stores `rollout_len` steps over `num_envs` envs and computes GAE."""

    def __init__(self, num_envs: int, rollout_len: int, obs_dim: int, act_dim: int,
                 device: torch.device) -> None:
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.device = device
        self.obs = torch.zeros(rollout_len, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(rollout_len, num_envs, act_dim, device=device)
        self.logp = torch.zeros(rollout_len, num_envs, device=device)
        self.values = torch.zeros(rollout_len, num_envs, device=device)
        self.rewards = torch.zeros(rollout_len, num_envs, device=device)
        self.dones = torch.zeros(rollout_len, num_envs, device=device)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(self, obs, action, logp, value, reward, done) -> None:
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.logp[t] = logp
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done.float() if done.dtype == torch.bool else done
        self.ptr += 1

    def compute_gae(
        self, last_value: torch.Tensor, gamma: float, lam: float,
    ) -> RolloutData:
        T = self.rollout_len
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * non_terminal - self.values[t]
            gae = delta + gamma * lam * non_terminal * gae
            adv[t] = gae
        returns = adv + self.values
        return RolloutData(
            obs=self.obs.clone(), actions=self.actions.clone(),
            logp=self.logp.clone(), values=self.values.clone(),
            rewards=self.rewards.clone(), dones=self.dones.clone(),
            advantages=adv, returns=returns,
        )
