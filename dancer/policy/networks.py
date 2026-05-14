"""Actor / Critic MLP networks for PPO."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    return {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[name.lower()]()


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

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, greedy: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, logp, value). Used during rollout collection."""
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
