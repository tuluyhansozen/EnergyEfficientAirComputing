"""Actor-Critic agent implementation.

This module provides Actor-Critic algorithms including A2C.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from aircompsim.drl.base import BaseAgent

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network.

    Outputs both policy (actor) and value (critic) estimates.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256) -> None:
        """Initialize network.

        Args:
            state_size: Input dimension.
            action_size: Number of actions.
            hidden_size: Hidden layer size.
        """
        super().__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)

        # Critic head (value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: Input state tensor.

        Returns:
            Tuple of (action_probs, state_value).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor output: action probabilities
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic output: state value
        state_value = self.critic(x)

        return action_probs, state_value


class ActorCriticAgent(BaseAgent):
    """Actor-Critic agent with advantage estimation.

    Implements the Advantage Actor-Critic (A2C) algorithm.

    Example:
        >>> agent = ActorCriticAgent(state_size=4, action_size=5)
        >>> action = agent.select_action(state)
        >>> agent.learn(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        hidden_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """Initialize Actor-Critic agent.

        Args:
            state_size: Dimension of state space.
            action_size: Number of actions.
            learning_rate: Learning rate.
            discount_factor: Discount factor (gamma).
            hidden_size: Hidden layer size.
            entropy_coef: Entropy bonus coefficient.
            value_coef: Value loss coefficient.
            device: Computation device.
        """
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            device=device,
        )

        self.hidden_size = hidden_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Initialize network
        self.network = ActorCriticNetwork(
            state_size=state_size, action_size=action_size, hidden_size=hidden_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # For storing policy info
        self.log_prob: Optional[torch.Tensor] = None

        # Metrics
        self.actor_losses: list[float] = []
        self.critic_losses: list[float] = []

        logger.info(
            f"ActorCriticAgent initialized: state_size={state_size}, " f"action_size={action_size}"
        )

    def select_action(self, state: np.ndarray) -> int:
        """Select action using policy network.

        Args:
            state: Current state observation.

        Returns:
            Sampled action.
        """
        self.network.eval()
        state_tensor = torch.from_numpy(state).float().to(self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad() if not self.training else torch.enable_grad():
            action_probs, _ = self.network(state_tensor)

            if self.training:
                # Sample action from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                self.log_prob = dist.log_prob(action)
                return action.item()
            else:
                # Take greedy action
                return action_probs.argmax().item()

    def learn(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> Optional[float]:
        """Update agent using TD error.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode ended.

        Returns:
            Total loss.
        """
        if self.log_prob is None:
            return None

        self.network.train()
        self.optimizer.zero_grad()

        # Convert to tensors
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        next_state_t = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # Get current and next values
        _, current_value = self.network(state_t)
        _, next_value = self.network(next_state_t)

        # TD target
        target = reward_t + self.discount_factor * next_value * (1 - int(done))

        # Advantage
        advantage = target - current_value

        # Actor loss (policy gradient)
        actor_loss = -self.log_prob * advantage.detach()

        # Critic loss (TD error)
        critic_loss = F.mse_loss(current_value, target.detach())

        # Entropy bonus for exploration
        action_probs, _ = self.network(state_t)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()

        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        total_loss.backward()
        self.optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.total_steps += 1

        # Reset log_prob for next action
        self.log_prob = None

        return total_loss.item()

    def save(self, path: str) -> None:
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "config": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                },
            },
            path,
        )

        logger.info(f"ActorCriticAgent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)

        logger.info(f"ActorCriticAgent loaded from {path}")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "total_steps": self.total_steps,
            "avg_actor_loss": np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            "avg_critic_loss": np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
        }
