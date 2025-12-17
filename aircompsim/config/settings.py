"""Configuration management.

This module provides dataclasses for simulation configuration
and YAML/JSON config loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BoundaryConfig:
    """Simulation boundary configuration."""

    max_x: float = 400.0
    max_y: float = 400.0
    max_z: float = 400.0
    min_x: float = 0.0
    min_y: float = 0.0
    min_z: float = 0.0


@dataclass
class EdgeServerConfig:
    """Edge server configuration."""

    capacity: float = 1000.0
    radius: float = 100.0
    power: float = 100.0
    count: int = 4
    locations: Optional[List[tuple]] = None


@dataclass
class UAVConfig:
    """UAV configuration."""

    capacity: float = 500.0
    radius: float = 100.0
    power: float = 50.0
    count: int = 5
    altitude: float = 200.0
    initial_battery: float = 100.0
    speed: float = 2.5


@dataclass
class CloudServerConfig:
    """Cloud server configuration."""

    capacity: float = 100000.0
    network_delay: float = 1.5


@dataclass
class EnergyConfig:
    """Energy model configuration."""

    alpha: float = 0.05  # Flight coefficient
    beta: float = 3.0  # Hover power
    gamma: float = 20.0  # Computation power
    delta: float = 10.0  # Communication power
    use_physics_model: bool = False
    low_battery_threshold: float = 30.0
    critical_battery_threshold: float = 10.0


@dataclass
class ApplicationConfig:
    """Application type configuration."""

    name: str = "Default"
    cpu_cycle: float = 100.0
    worst_delay: float = 1.0
    best_delay: float = 0.1
    interarrival_time: float = 10.0


@dataclass
class DRLConfig:
    """Deep Reinforcement Learning configuration."""

    enabled: bool = False
    algorithm: str = "DDQN"  # DQN, DDQN, ActorCritic
    learning_rate: float = 0.0001
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99
    batch_size: int = 64
    memory_size: int = 100000
    episodes: int = 500
    state_interval: int = 3


@dataclass
class MobilityConfig:
    """Mobility configuration."""

    user_mobility: bool = True
    user_speed: float = 2.0  # m/s
    uav_fly_policy: str = "LSI"  # LSI, Random, DRL, NoUAV
    uav_waiting_policy: float = 100.0  # seconds


@dataclass
class SimulationConfig:
    """Main simulation configuration.

    Aggregates all sub-configurations for the simulation.

    Example:
        >>> config = SimulationConfig()
        >>> config.time_limit = 2000
        >>> config.uav.count = 10
    """

    # Simulation parameters
    time_limit: float = 1000.0
    warmup_period: float = 100.0
    seed: Optional[int] = None
    log_level: str = "INFO"
    output_dir: str = "results"

    # User parameters
    user_count: int = 20

    # Sub-configurations
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    edge: EdgeServerConfig = field(default_factory=EdgeServerConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    cloud: CloudServerConfig = field(default_factory=CloudServerConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    drl: DRLConfig = field(default_factory=DRLConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    applications: List[ApplicationConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set up default applications if none provided."""
        if not self.applications:
            self.applications = [
                ApplicationConfig(
                    name="Entertainment",
                    cpu_cycle=100,
                    worst_delay=0.3,
                    best_delay=0.1,
                    interarrival_time=10,
                ),
                ApplicationConfig(
                    name="Multimedia",
                    cpu_cycle=100,
                    worst_delay=3.0,
                    best_delay=0.1,
                    interarrival_time=10,
                ),
                ApplicationConfig(
                    name="Rendering",
                    cpu_cycle=200,
                    worst_delay=1.0,
                    best_delay=0.5,
                    interarrival_time=20,
                ),
                ApplicationConfig(
                    name="ImageClassification",
                    cpu_cycle=600,
                    worst_delay=1.0,
                    best_delay=0.5,
                    interarrival_time=20,
                ),
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses

        def convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        return dict(convert(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            SimulationConfig instance.
        """
        config = cls()

        # Simple fields
        for key in ["time_limit", "warmup_period", "seed", "log_level", "output_dir", "user_count"]:
            if key in data:
                setattr(config, key, data[key])

        # Nested configs
        if "boundary" in data:
            config.boundary = BoundaryConfig(**data["boundary"])
        if "edge" in data:
            config.edge = EdgeServerConfig(**data["edge"])
        if "uav" in data:
            config.uav = UAVConfig(**data["uav"])
        if "cloud" in data:
            config.cloud = CloudServerConfig(**data["cloud"])
        if "energy" in data:
            config.energy = EnergyConfig(**data["energy"])
        if "drl" in data:
            config.drl = DRLConfig(**data["drl"])
        if "mobility" in data:
            config.mobility = MobilityConfig(**data["mobility"])
        if "applications" in data:
            config.applications = [ApplicationConfig(**app) for app in data["applications"]]

        return config


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration (e.g., for different scenarios)."""

    name: str = "default"
    description: str = ""
    config: SimulationConfig = field(default_factory=SimulationConfig)
