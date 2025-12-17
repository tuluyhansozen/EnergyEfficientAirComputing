"""Energy-aware task scheduling.

This module provides schedulers that consider energy efficiency
when making task offloading and UAV deployment decisions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from aircompsim.entities.server import Server, EdgeServer, UAV, CloudServer
    from aircompsim.entities.task import Task
    from aircompsim.entities.user import User

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Task scheduling strategies.

    Attributes:
        ENERGY_FIRST: Minimize energy consumption.
        LATENCY_FIRST: Minimize task latency.
        BALANCED: Balance energy and latency.
        UTILIZATION: Balance server utilization.
    """

    ENERGY_FIRST = auto()
    LATENCY_FIRST = auto()
    BALANCED = auto()
    UTILIZATION = auto()


class ServerProtocol(Protocol):
    """Protocol for server-like objects."""

    @property
    def id(self) -> int: ...

    @property
    def capacity(self) -> float: ...

    @property
    def next_available_time(self) -> float: ...

    def get_processing_delay(self, task: Task) -> float: ...

    def is_in_coverage(self, location) -> bool: ...


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision.

    Attributes:
        server: Selected server for task.
        estimated_latency: Expected task completion time.
        estimated_energy: Expected energy consumption.
        confidence: Decision confidence (0-1).
        reason: Human-readable decision rationale.
    """

    server: Optional[Server]
    estimated_latency: float = 0.0
    estimated_energy: float = 0.0
    confidence: float = 1.0
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if a valid server was selected."""
        return self.server is not None


class BaseScheduler(ABC):
    """Abstract base class for task schedulers.

    Subclasses implement specific scheduling algorithms.
    """

    @abstractmethod
    def select_server(
        self, task: Task, available_servers: list[Server], current_time: float
    ) -> SchedulingDecision:
        """Select optimal server for a task.

        Args:
            task: Task to be scheduled.
            available_servers: List of available servers.
            current_time: Current simulation time.

        Returns:
            SchedulingDecision with selected server and metrics.
        """
        pass

    @abstractmethod
    def should_accept_task(self, server: Server, task: Task) -> bool:
        """Determine if a server should accept a task.

        Args:
            server: Server being considered.
            task: Task to be accepted.

        Returns:
            True if server should accept the task.
        """
        pass


class EnergyAwareScheduler(BaseScheduler):
    """Scheduler that optimizes for energy efficiency.

    Considers energy costs of computation, communication,
    and UAV operations when making scheduling decisions.

    Attributes:
        strategy: Scheduling strategy to use.
        energy_weight: Weight for energy in balanced mode (0-1).
        latency_weight: Weight for latency in balanced mode (0-1).

    Example:
        >>> scheduler = EnergyAwareScheduler(strategy=SchedulingStrategy.BALANCED)
        >>> decision = scheduler.select_server(task, servers, current_time)
        >>> if decision.is_valid:
        ...     task.assigned_server = decision.server
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.BALANCED,
        energy_weight: float = 0.5,
        latency_weight: float = 0.5,
        uav_energy_threshold: float = 20.0,
    ) -> None:
        """Initialize scheduler.

        Args:
            strategy: Scheduling strategy.
            energy_weight: Weight for energy in cost function.
            latency_weight: Weight for latency in cost function.
            uav_energy_threshold: Minimum UAV battery for task acceptance.
        """
        self.strategy = strategy
        self.energy_weight = energy_weight
        self.latency_weight = latency_weight
        self.uav_energy_threshold = uav_energy_threshold

        # Normalize weights
        total = self.energy_weight + self.latency_weight
        if total > 0:
            self.energy_weight /= total
            self.latency_weight /= total

        logger.info(
            f"EnergyAwareScheduler initialized: strategy={strategy.name}, "
            f"energy_weight={self.energy_weight:.2f}, latency_weight={self.latency_weight:.2f}"
        )

    def select_server(
        self, task: Task, available_servers: list[Server], current_time: float
    ) -> SchedulingDecision:
        """Select optimal server considering energy efficiency.

        Args:
            task: Task to be scheduled.
            available_servers: List of available servers.
            current_time: Current simulation time.

        Returns:
            SchedulingDecision with selected server.
        """
        if not available_servers:
            logger.warning("No available servers for scheduling")
            return SchedulingDecision(server=None, reason="No available servers")

        # Filter servers that can accept the task
        valid_servers = [s for s in available_servers if self.should_accept_task(s, task)]

        if not valid_servers:
            logger.warning("No servers can accept the task")
            return SchedulingDecision(server=None, reason="No servers can accept task")

        # Calculate cost for each server
        best_server = None
        best_cost = float("inf")
        best_metrics = {"latency": 0, "energy": 0}

        for server in valid_servers:
            latency = self._estimate_latency(server, task, current_time)
            energy = self._estimate_energy(server, task)

            cost = self._calculate_cost(latency, energy, task)

            if cost < best_cost:
                best_cost = cost
                best_server = server
                best_metrics = {"latency": latency, "energy": energy}

        if best_server is None:
            return SchedulingDecision(server=None, reason="Could not find suitable server")

        server_type = type(best_server).__name__
        decision = SchedulingDecision(
            server=best_server,
            estimated_latency=best_metrics["latency"],
            estimated_energy=best_metrics["energy"],
            confidence=self._calculate_confidence(best_metrics, task),
            reason=f"Selected {server_type} {best_server.id} (cost={best_cost:.3f})",
        )

        logger.debug(
            f"Scheduling decision: {decision.reason}, "
            f"latency={decision.estimated_latency:.3f}s, "
            f"energy={decision.estimated_energy:.3f}J"
        )

        return decision

    def should_accept_task(self, server: Server, task: Task) -> bool:
        """Check if server should accept task based on energy state.

        Args:
            server: Server being considered.
            task: Task to be accepted.

        Returns:
            True if server should accept the task.
        """
        # Import here to avoid circular imports
        from aircompsim.entities.server import UAV

        # For UAVs, check battery level
        if isinstance(server, UAV):
            if hasattr(server, "battery_level"):
                if server.battery_level < self.uav_energy_threshold:
                    logger.debug(
                        f"UAV {server.id} rejected task: "
                        f"battery {server.battery_level:.1f}% < threshold"
                    )
                    return False

            if hasattr(server, "energy_mode"):
                from aircompsim.energy.models import EnergyMode

                if server.energy_mode == EnergyMode.CRITICAL:
                    logger.debug(f"UAV {server.id} rejected task: CRITICAL energy mode")
                    return False

        return True

    def _estimate_latency(self, server: Server, task: Task, current_time: float) -> float:
        """Estimate task completion latency for a server.

        Args:
            server: Server to estimate for.
            task: Task to be processed.
            current_time: Current simulation time.

        Returns:
            Estimated latency in seconds.
        """
        # Queueing delay
        queue_delay = max(0, server.next_available_time - current_time)

        # Processing delay
        processing_delay = server.get_processing_delay(task)

        # Network delay (simplified)
        from aircompsim.entities.server import EdgeServer, UAV, CloudServer

        if isinstance(server, CloudServer):
            network_delay = 1.5  # WAN delay
        elif isinstance(server, UAV):
            network_delay = 0.005  # Air-to-ground
        else:
            network_delay = 0.001  # LAN delay

        return queue_delay + processing_delay + network_delay

    def _estimate_energy(self, server: Server, task: Task) -> float:
        """Estimate energy consumption for processing a task.

        Args:
            server: Server to process task.
            task: Task to be processed.

        Returns:
            Estimated energy in Joules.
        """
        from aircompsim.entities.server import EdgeServer, UAV, CloudServer

        # Base computation energy
        processing_time = server.get_processing_delay(task)

        if isinstance(server, UAV):
            # UAV: computation + communication + hover
            computation_power = 20  # W
            communication_power = 10  # W
            hover_power = 3  # W
            energy = (computation_power + communication_power) * processing_time
            energy += hover_power * processing_time  # Hover during processing
        elif isinstance(server, EdgeServer):
            # Edge: just computation
            computation_power = 50  # W (edge servers use more power per computation)
            energy = computation_power * processing_time
        else:
            # Cloud: high power but shared
            computation_power = 100  # W
            energy = computation_power * processing_time * 0.1  # Shared resources

        return energy

    def _calculate_cost(self, latency: float, energy: float, task: Task) -> float:
        """Calculate combined cost for a scheduling decision.

        Args:
            latency: Estimated latency.
            energy: Estimated energy.
            task: Task being scheduled.

        Returns:
            Combined cost value (lower is better).
        """
        # Normalize values
        max_latency = task.app.worst_delay if hasattr(task.app, "worst_delay") else 1.0
        latency_norm = latency / max_latency if max_latency > 0 else latency

        # Energy normalization (assuming 1000J as reference)
        energy_norm = energy / 1000.0

        if self.strategy == SchedulingStrategy.ENERGY_FIRST:
            return energy_norm
        elif self.strategy == SchedulingStrategy.LATENCY_FIRST:
            return latency_norm
        else:  # BALANCED
            return self.energy_weight * energy_norm + self.latency_weight * latency_norm

    def _calculate_confidence(self, metrics: dict, task: Task) -> float:
        """Calculate confidence in scheduling decision.

        Args:
            metrics: Decision metrics (latency, energy).
            task: Task being scheduled.

        Returns:
            Confidence value (0-1).
        """
        # Base confidence
        confidence = 0.8

        # Reduce confidence if latency is close to deadline
        if hasattr(task.app, "worst_delay"):
            deadline_ratio = metrics["latency"] / task.app.worst_delay
            if deadline_ratio > 0.9:
                confidence *= 0.5
            elif deadline_ratio > 0.7:
                confidence *= 0.8

        return min(1.0, confidence)


@dataclass
class SchedulingMetrics:
    """Metrics for evaluating scheduling performance.

    Attributes:
        total_tasks: Total tasks scheduled.
        successful_tasks: Tasks completed within deadline.
        total_energy: Total energy consumed.
        average_latency: Average task latency.
        server_utilization: Utilization by server type.
    """

    total_tasks: int = 0
    successful_tasks: int = 0
    total_energy: float = 0.0
    total_latency: float = 0.0
    edge_tasks: int = 0
    uav_tasks: int = 0
    cloud_tasks: int = 0

    @property
    def success_rate(self) -> float:
        """Get task success rate."""
        return self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def average_latency(self) -> float:
        """Get average task latency."""
        return self.total_latency / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def average_energy(self) -> float:
        """Get average energy per task."""
        return self.total_energy / self.total_tasks if self.total_tasks > 0 else 0.0

    def record_task(self, server_type: str, latency: float, energy: float, success: bool) -> None:
        """Record a completed task.

        Args:
            server_type: Type of server that processed task.
            latency: Task latency.
            energy: Energy consumed.
            success: Whether task met deadline.
        """
        self.total_tasks += 1
        self.total_latency += latency
        self.total_energy += energy

        if success:
            self.successful_tasks += 1

        if server_type == "EdgeServer":
            self.edge_tasks += 1
        elif server_type == "UAV":
            self.uav_tasks += 1
        else:
            self.cloud_tasks += 1

    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            "total_tasks": self.total_tasks,
            "success_rate": self.success_rate,
            "average_latency": self.average_latency,
            "average_energy": self.average_energy,
            "total_energy": self.total_energy,
            "distribution": {
                "edge": self.edge_tasks,
                "uav": self.uav_tasks,
                "cloud": self.cloud_tasks,
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_energy = 0.0
        self.total_latency = 0.0
        self.edge_tasks = 0
        self.uav_tasks = 0
        self.cloud_tasks = 0
