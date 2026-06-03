"""Actor / Critic MLP networks for PPO."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    return {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[name.lower()]()


class RunningMeanStd(nn.Module):
    """Tracks a running mean/variance of observations (Welford, batched).

    Stats live in registered buffers so they are saved in the checkpoint and
    restored automatically at eval time. Updated only during rollout
    collection (between PPO updates), never inside the optimisation epochs,
    so that within one iteration the normalisation is fixed and the
    first-minibatch importance ratio stays ≈ 1.
    """

    def __init__(self, dim: int, epsilon: float = 1e-4, clip: float = 5.0) -> None:
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return x.clamp(-self.clip, self.clip)


def _orthogonal_init(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int],
        out_dim: int,
        activation: str = "elu",
        final_gain: float = 0.01,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_sizes, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            gain = final_gain if i == len(dims) - 2 else math.sqrt(2.0)
            _orthogonal_init(lin, gain=gain)
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """Gaussian policy with optional learnable log_std (state-independent)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (1024, 1024),
        activation: str = "elu",
        log_std_init: float = -0.7,         # log(0.5)
        learnable_log_std: bool = False,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.mu = MLP(obs_dim, hidden_sizes, act_dim, activation, final_gain=0.01)
        log_std = torch.full((act_dim,), float(log_std_init))
        if learnable_log_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mu(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (1024, 1024),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.v = MLP(obs_dim, hidden_sizes, 1, activation, final_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (1024, 1024),
        activation: str = "elu",
        log_std_init: float = -0.7,
        learnable_log_std: bool = False,
    ) -> None:
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_sizes, activation,
                           log_std_init, learnable_log_std)
        self.critic = Critic(obs_dim, hidden_sizes, activation)
        self.obs_rms = RunningMeanStd(obs_dim)

    def update_obs_rms(self, obs: torch.Tensor) -> None:
        """Fold a batch of raw observations into the running normaliser.

        Call once per iteration from the training loop (on the whole rollout),
        not during PPO epochs — see RunningMeanStd docstring.
        """
        self.obs_rms.update(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, greedy: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, logp, value). Used during rollout collection."""
        obs = self.obs_rms.normalize(obs)
        mean, std = self.actor(obs)
        if greedy:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)
        logp = self._gaussian_logp(action, mean, std)
        value = self.critic(obs)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logp, value, entropy) for given (obs, action). For PPO updates."""
        obs = self.obs_rms.normalize(obs)
        mean, std = self.actor(obs)
        logp = self._gaussian_logp(action, mean, std)
        entropy = 0.5 * (1.0 + math.log(2.0 * math.pi)) * self.actor.act_dim \
                  + std.log().sum(dim=-1)
        value = self.critic(obs)
        return logp, value, entropy

    @staticmethod
    def _gaussian_logp(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
                       ) -> torch.Tensor:
        var = std * std
        return (-0.5 * (((action - mean) ** 2) / var + 2.0 * std.log()
                         + math.log(2.0 * math.pi))).sum(dim=-1)
