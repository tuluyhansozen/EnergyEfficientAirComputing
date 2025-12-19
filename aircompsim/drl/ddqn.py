"""Double Deep Q-Network (DDQN) agent implementation.

This module provides the DDQN algorithm which addresses
overestimation bias in vanilla DQN.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aircompsim.drl.base import BaseAgent, QNetwork, ReplayBuffer

logger = logging.getLogger(__name__)


class DDQNAgent(BaseAgent):
    """Double Deep Q-Network agent.

    Implements DDQN with separate online and target networks
    to reduce overestimation of Q-values.

    Attributes:
        network: Online Q-Network.
        target_network: Target Q-Network.
        optimizer: Network optimizer.
        replay_buffer: Experience replay buffer.

    Example:
        >>> agent = DDQNAgent(state_size=4, action_size=5)
        >>> action = agent.select_action(state)
        >>> agent.learn(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        hidden_size: int = 128,
        buffer_size: int = 100000,
        batch_size: int = 64,
        tau: float = 0.01,
        target_update_freq: int = 10,
        device: str | None = None,
    ) -> None:
        """Initialize DDQN agent.

        Args:
            state_size: Dimension of state space.
            action_size: Number of actions.
            learning_rate: Learning rate.
            discount_factor: Discount factor (gamma).
            hidden_size: Hidden layer size.
            buffer_size: Replay buffer capacity.
            batch_size: Training batch size.
            tau: Soft update coefficient.
            target_update_freq: Steps between target updates.
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
        self.tau = tau
        self.target_update_freq = target_update_freq

        # Initialize networks
        self.network = QNetwork(
            state_size=state_size, action_size=action_size, hidden_size=hidden_size
        ).to(self.device)

        self.target_network = QNetwork(
            state_size=state_size, action_size=action_size, hidden_size=hidden_size
        ).to(self.device)

        # Copy weights to target network
        self._hard_update()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Metrics
        self.losses: list[float] = []
        self._update_counter = 0

        logger.info(
            f"DDQNAgent initialized: state_size={state_size}, "
            f"action_size={action_size}, tau={tau}"
        )

    def _hard_update(self) -> None:
        """Copy online network weights to target network."""
        self.target_network.load_state_dict(self.network.state_dict())

    def _soft_update(self) -> None:
        """Soft update target network toward online network."""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
            action = int(q_values.argmax().item())
        return action

    def learn(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> float | None:
        """Update agent from single experience.

        Uses Double Q-learning: online network selects action,
        target network evaluates value.

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
        """Perform single training step with DDQN update.

        Returns:
            Training loss.
        """
        self.network.train()

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        state_tensor = torch.from_numpy(states).float().to(self.device)
        action_tensor = torch.from_numpy(actions).long().to(self.device)
        reward_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_state_tensor = torch.from_numpy(next_states).float().to(self.device)
        done_tensor = torch.from_numpy(dones).float().to(self.device)

        # Current Q-values
        current_q = self.network(state_tensor)
        current_q = current_q.gather(1, action_tensor.unsqueeze(1)).squeeze()

        # DDQN: Online network selects action, target network evaluates
        with torch.no_grad():
            # Online network selects best action
            next_q_online = self.network(next_state_tensor)
            best_actions = next_q_online.argmax(dim=1)

            # Target network evaluates selected action
            next_q_target = self.target_network(next_state_tensor)
            next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze()

            # Compute target
            target_q = reward_tensor + self.discount_factor * next_q_values * (1 - done_tensor)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self._update_counter += 1

        # Soft update target network
        self._soft_update()

        # Decay epsilon
        self.update_epsilon()

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save agent to file.

        Args:
            path: Save path.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "config": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "tau": self.tau,
                },
            },
            save_path,
        )

        logger.info(f"DDQNAgent saved to {save_path}")

    def load(self, path: str) -> None:
        """Load agent from file.

        Args:
            path: Load path.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", 0.01)
        self.total_steps = checkpoint.get("total_steps", 0)

        logger.info(f"DDQNAgent loaded from {path}")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "update_counter": self._update_counter,
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0.0,
        }
