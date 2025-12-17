"""Deep Q-Network (DQN) agent implementation.

This module provides the DQN algorithm for discrete action spaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aircompsim.drl.base import BaseAgent, QNetwork, ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent.

    Implements the DQN algorithm with experience replay.

    Attributes:
        network: Q-Network for action-value estimation.
        optimizer: Network optimizer.
        replay_buffer: Experience replay buffer.

    Example:
        >>> agent = DQNAgent(state_size=4, action_size=5)
        >>> action = agent.select_action(state)
        >>> agent.learn(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        hidden_size: int = 128,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        """Initialize DQN agent.

        Args:
            state_size: Dimension of state space.
            action_size: Number of actions.
            learning_rate: Learning rate.
            discount_factor: Discount factor (gamma).
            hidden_size: Hidden layer size.
            buffer_size: Replay buffer capacity.
            batch_size: Training batch size.
            device: Computation device.
        """
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            device=device,
        )

        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # Initialize network
        self.network = QNetwork(
            state_size=state_size, action_size=action_size, hidden_size=hidden_size
        ).to(self.device)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Metrics
        self.losses: list[float] = []

        logger.info(
            f"DQNAgent initialized: state_size={state_size}, "
            f"action_size={action_size}, hidden={hidden_size}"
        )

    def select_action(self, state: np.ndarray) -> int:
        """Select best action based on Q-values.

        Args:
            state: Current state observation.

        Returns:
            Action with highest Q-value.
        """
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            q_values = self.network(state_tensor)
            action = q_values.argmax().item()
        return action

    def learn(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> Optional[float]:
        """Update agent from single experience.

        Stores experience and trains if buffer is ready.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode ended.

        Returns:
            Training loss if training occurred.
        """
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Train if buffer is ready
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        return self._train_step()

    def _train_step(self) -> float:
        """Perform single training step.

        Returns:
            Training loss.
        """
        self.network.train()

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Current Q-values
        current_q = self.network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values (using same network - vanilla DQN)
        with torch.no_grad():
            next_q = self.network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.discount_factor * max_next_q * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        # Decay epsilon
        self.update_epsilon()

        return loss.item()

    def save(self, path: str) -> None:
        """Save agent to file.

        Args:
            path: Save path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
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

        logger.info(f"DQNAgent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent from file.

        Args:
            path: Load path.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", 0.01)
        self.total_steps = checkpoint.get("total_steps", 0)

        logger.info(f"DQNAgent loaded from {path}")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0.0,
        }
