"""Base classes for Deep Reinforcement Learning agents.

This module provides abstract base classes and common utilities
for DRL agent implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple for replay buffer.
    
    Attributes:
        state: Current state observation.
        action: Action taken.
        reward: Reward received.
        next_state: Next state observation.
        done: Whether episode ended.
    """
    
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool = False


class ReplayBuffer:
    """Experience replay buffer for DRL training.
    
    Stores experiences and provides random sampling for training.
    
    Attributes:
        capacity: Maximum buffer size.
        
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=64)
    """
    
    def __init__(self, capacity: int = 100000) -> None:
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store.
        """
        self.capacity = capacity
        self._buffer: List[Experience] = []
        self._position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ) -> None:
        """Add an experience to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode ended.
        """
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
        else:
            self._buffer[self._position] = experience
        
        self._position = (self._position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        import random
        
        batch_size = min(batch_size, len(self._buffer))
        batch = random.sample(self._buffer, batch_size)
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self._buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self._buffer.clear()
        self._position = 0


class BaseAgent(ABC):
    """Abstract base class for DRL agents.
    
    Provides common interface and utilities for all agent types.
    
    Attributes:
        state_size: Dimension of state space.
        action_size: Number of possible actions.
        device: PyTorch device for computation.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        device: Optional[str] = None
    ) -> None:
        """Initialize base agent.
        
        Args:
            state_size: Dimension of state observation.
            action_size: Number of discrete actions.
            learning_rate: Learning rate for optimizer.
            discount_factor: Discount factor (gamma).
            device: Device for computation ('cpu', 'cuda', 'mps').
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        logger.info(f"Agent initialized on device: {self.device}")
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Training state
        self.training = True
        self.total_steps = 0
        self.episode_rewards: List[float] = []
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Select action given current state.
        
        Args:
            state: Current state observation.
            
        Returns:
            Selected action index.
        """
        pass
    
    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """Update agent from experience.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode ended.
            
        Returns:
            Training loss if applicable.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to file.
        
        Args:
            path: Save path.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from file.
        
        Args:
            path: Load path.
        """
        pass
    
    def update_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def set_training(self, mode: bool) -> None:
        """Set training mode.
        
        Args:
            mode: True for training, False for evaluation.
        """
        self.training = mode
    
    def get_action_exploration(self, state: np.ndarray) -> int:
        """Select action with epsilon-greedy exploration.
        
        Args:
            state: Current state.
            
        Returns:
            Selected action.
        """
        if self.training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        return self.select_action(state)


class QNetwork(nn.Module):
    """Simple Q-Network for value-based DRL.
    
    Three-layer fully connected network.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128
    ) -> None:
        """Initialize Q-Network.
        
        Args:
            state_size: Input dimension.
            action_size: Output dimension.
            hidden_size: Hidden layer size.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Q-values for each action.
        """
        return self.network(x)


@dataclass
class TrainingConfig:
    """Configuration for DRL training.
    
    Attributes:
        episodes: Number of training episodes.
        batch_size: Training batch size.
        target_update: Steps between target network updates.
        save_frequency: Episodes between model saves.
    """
    
    episodes: int = 500
    batch_size: int = 64
    target_update: int = 10
    save_frequency: int = 100
    learning_rate: float = 0.0001
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
