"""
Actor-Critic agent for 6x6 Checkers.

Uses a simple shared MLP with separate actor + critic outputs.
Trained with Monte Carlo returns from full episodes (self-play).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mycheckersenv import OBS_SIZE, NUM_ACTIONS

GAMMA = 0.99
LR = 3e-4
ENTROPY_COEF = 0.01
CRITIC_COEF  = 0.5
HIDDEN_SIZE  = 128


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)
        self.critic_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        features = self.trunk(x)
        return self.actor_head(features), self.critic_head(features)


class ActorCriticAgent:
    def __init__(self):
        self.net       = ActorCriticNet()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._entropies = []

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t  = torch.FloatTensor(obs.astype(np.float32) / 4.0)
        mask_t = torch.BoolTensor(mask.astype(bool))

        logits, value = self.net(obs_t)
        logits[~mask_t] = float("-inf")

        dist   = Categorical(logits=logits)
        action = dist.sample()

        self._log_probs.append(dist.log_prob(action))
        self._values.append(value.squeeze())
        self._entropies.append(dist.entropy())

        return action.item()

    def store(self, reward: float):
        self._rewards.append(float(reward))

    def reset_trajectory(self):
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._entropies = []

    def finish_episode(self) -> dict:
        T = len(self._rewards)
        if T == 0:
            return {}

        # compute discounted returns backwards
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + GAMMA * G
            returns[t] = G

        log_probs  = torch.stack(self._log_probs)
        values     = torch.stack(self._values)
        entropies  = torch.stack(self._entropies)
        advantages = returns.detach() - values

        actor_loss  = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -entropies.mean()
        total_loss  = actor_loss + CRITIC_COEF * critic_loss + ENTROPY_COEF * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.reset_trajectory()

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     -entropy_loss.item(),
            "mean_return": returns.mean().item(),
        }

    def select_greedy_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs.astype(np.float32) / 4.0)
            mask_t = torch.BoolTensor(mask.astype(bool))
            logits, _ = self.net(obs_t)
            logits[~mask_t] = float("-inf")
            return logits.argmax().item()

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, path)
        print(f"Saved checkpoint → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded checkpoint ← {path}")