"""Server entity definitions.

This module provides server classes including EdgeServer, UAV, and CloudServer
for modeling computational resources in air computing environments.
"""

from __future__ import annotations

import logging
import math
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, ClassVar

if TYPE_CHECKING:
    from aircompsim.entities.task import Task, Application
    from aircompsim.entities.user import User

from aircompsim.entities.location import Location
from aircompsim.energy.models import EnergyModel, EnergyMode, EnergyTracker

logger = logging.getLogger(__name__)


@dataclass
class Server(ABC):
    """Abstract base class for all server types.
    
    Provides common functionality for computational servers including
    coverage checking, utilization tracking, and task processing.
    
    Attributes:
        capacity: Computational capacity in cycles/second.
        location: Server location.
        radius: Coverage radius in meters.
        power_consumption: Power consumption in Watts.
        server_id: Unique server identifier.
        
    Note:
        This is an abstract class. Use EdgeServer, UAV, or CloudServer.
    """
    
    capacity: float
    location: Location
    radius: float
    power_consumption: float
    server_id: int = field(default=0, repr=False)
    
    # State tracking
    utilization: float = field(default=0.0, repr=False)
    next_available_time: float = field(default=0.0, repr=False)
    _inner_time: float = field(default=0.0, repr=False)
    _processed_tasks: List = field(default_factory=list, repr=False)
    
    # Energy tracking
    energy_tracker: EnergyTracker = field(default_factory=EnergyTracker, repr=False)
    energy_model: EnergyModel = field(default_factory=EnergyModel, repr=False)
    
    def is_in_coverage(self, loc: Location) -> bool:
        """Check if a location is within server coverage.
        
        Args:
            loc: Location to check.
            
        Returns:
            True if location is within coverage radius.
        """
        distance = Location.euclidean_distance_2d(self.location, loc)
        return distance <= self.radius
    
    def get_utilization(self, time_limit: float) -> float:
        """Get server utilization percentage.
        
        Args:
            time_limit: Total simulation time.
            
        Returns:
            Utilization as percentage (0-100).
        """
        if self._inner_time < time_limit:
            self._inner_time = time_limit
        
        if self._inner_time == 0:
            return 0.0
        
        utilization_pct = (self.utilization / self._inner_time) * 100
        utilization_pct = min(100.0, utilization_pct)
        
        logger.debug(f"Server {self.server_id} utilization: {utilization_pct:.1f}%")
        return utilization_pct
    
    def get_instant_utilization(
        self,
        time_interval: float,
        sim_time: float
    ) -> float:
        """Get utilization over a recent time window.
        
        Args:
            time_interval: Window size in seconds.
            sim_time: Current simulation time.
            
        Returns:
            Instantaneous utilization percentage.
        """
        if not self._processed_tasks:
            return 0.0
        
        max_time = sim_time - time_interval
        window_utilization = 0.0
        
        for task in reversed(self._processed_tasks):
            task_start = task.creation_time + task.waiting_time_in_queue
            if task_start > max_time and task_start < sim_time:
                window_utilization += task.processing_time
        
        return min(100.0, (window_utilization / time_interval) * 100)
    
    def get_processing_delay(self, task: Task) -> float:
        """Calculate processing delay for a task.
        
        Uses M/M/1 queueing model for delay calculation.
        
        Args:
            task: Task to process.
            
        Returns:
            Processing delay in seconds.
        """
        app = task.app
        mu = self.capacity / app.cpu_cycle  # Service rate
        processing_time = 1 / mu
        
        task.processing_time = processing_time
        self.utilization += processing_time
        self.next_available_time += processing_time
        self._inner_time += processing_time
        
        # Track energy
        energy = self.energy_model.compute_computation_energy(processing_time)
        self.energy_tracker.record(energy, "computation", self.next_available_time)
        
        logger.debug(
            f"Server {self.server_id}: Processing task {task.task_id}, "
            f"delay={processing_time:.4f}s"
        )
        
        return processing_time
    
    def update_processed_tasks(self, task: Task) -> None:
        """Record a processed task.
        
        Args:
            task: Completed task.
        """
        self._processed_tasks.append(task)
    
    def get_energy_consumption(self) -> float:
        """Get total energy consumed by server."""
        return self.energy_tracker.total_consumed
    
    def consume_communication_energy(self, duration: float) -> None:
        """Record communication energy consumption.
        
        Args:
            duration: Communication duration in seconds.
        """
        energy = self.energy_model.compute_communication_energy(duration)
        self.energy_tracker.record(energy, "communication")
        logger.debug(f"Server {self.server_id}: Comm energy {energy:.2f}J")
    
    def reset_stats(self) -> None:
        """Reset server statistics."""
        self.utilization = 0.0
        self.next_available_time = 0.0
        self._inner_time = 0.0
        self._processed_tasks.clear()
        self.energy_tracker.reset()


@dataclass
class EdgeServer(Server):
    """Fixed edge server at ground level.
    
    Edge servers have high capacity and are located in fixed positions.
    They provide low-latency processing for users within coverage.
    
    Attributes:
        All inherited from Server.
        
    Example:
        >>> edge = EdgeServer(
        ...     capacity=1000,
        ...     location=Location(100, 100, 0),
        ...     radius=100,
        ...     power_consumption=100
        ... )
    """
    
    # Class variables (not instance fields)
    _id_counter: ClassVar[int] = 0
    _all_servers: ClassVar[List[EdgeServer]] = []
    
    def __post_init__(self) -> None:
        """Initialize edge server with unique ID."""
        EdgeServer._id_counter += 1
        object.__setattr__(self, 'server_id', EdgeServer._id_counter)
        EdgeServer._all_servers.append(self)
        
        logger.info(
            f"EdgeServer {self.server_id} created at {self.location}, "
            f"capacity={self.capacity}, radius={self.radius}"
        )
    
    def __eq__(self, other: object) -> bool:
        """Check equality by ID."""
        if not isinstance(other, EdgeServer):
            return NotImplemented
        return self.server_id == other.server_id
    
    def __hash__(self) -> int:
        """Hash by ID."""
        return hash(("EdgeServer", self.server_id))
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all edge servers."""
        cls._id_counter = 0
        cls._all_servers.clear()
        logger.debug("EdgeServer registry reset")
    
    @classmethod
    def get_all(cls) -> List[EdgeServer]:
        """Get all edge servers."""
        return cls._all_servers.copy()
    
    @classmethod
    def remove(cls, server_id: int) -> bool:
        """Remove an edge server by ID."""
        for i, server in enumerate(cls._all_servers):
            if server.server_id == server_id:
                cls._all_servers.pop(i)
                logger.info(f"EdgeServer {server_id} removed")
                return True
        return False


@dataclass
class UAV(Server):
    """Unmanned Aerial Vehicle as flying edge server.
    
    UAVs have lower capacity than edge servers but provide mobile
    coverage. They track battery level and support energy-aware operations.
    
    Attributes:
        All inherited from Server.
        battery_level: Current battery percentage (0-100).
        altitude: Flying altitude in meters.
        horizontal_speed: Horizontal speed in m/s.
        vertical_speed: Vertical speed in m/s.
        is_flying: Whether UAV is currently in flight.
        flying_to: Destination location if flying.
        trajectory: History of visited locations.
        
    Example:
        >>> uav = UAV(
        ...     location=Location(50, 50, 200),
        ...     capacity=500,
        ...     radius=100,
        ...     power_consumption=50
        ... )
        >>> uav.consume_flight_energy(100, 10)  # 100m at 10m/s
    """
    
    battery_level: float = 100.0
    altitude: float = 200.0
    horizontal_speed: float = 0.0
    vertical_speed: float = 0.0
    is_flying: bool = False
    not_flying_since: float = 0.0
    flying_to: Optional[Location] = None
    trajectory: List[Location] = field(default_factory=list)
    
    # Class variables (not instance fields)
    _id_counter: ClassVar[int] = 0
    _all_uavs: ClassVar[List[UAV]] = []
    
    def __post_init__(self) -> None:
        """Initialize UAV with unique ID."""
        UAV._id_counter += 1
        object.__setattr__(self, 'server_id', UAV._id_counter)
        UAV._all_uavs.append(self)
        self.trajectory.append(self.location)
        
        logger.info(
            f"UAV {self.server_id} created at {self.location}, "
            f"battery={self.battery_level}%"
        )
    
    def __eq__(self, other: object) -> bool:
        """Check equality by ID."""
        if not isinstance(other, UAV):
            return NotImplemented
        return self.server_id == other.server_id
    
    def __hash__(self) -> int:
        """Hash by ID."""
        return hash(("UAV", self.server_id))
    
    @property
    def id(self) -> int:
        """Get UAV ID (alias for server_id)."""
        return self.server_id
    
    @property
    def energy_mode(self) -> EnergyMode:
        """Get current energy mode based on battery level."""
        if self.battery_level > 70:
            return EnergyMode.HIGH
        elif self.battery_level > 30:
            return EnergyMode.MEDIUM
        elif self.battery_level > 10:
            return EnergyMode.LOW
        else:
            return EnergyMode.CRITICAL
    
    @property
    def speed(self) -> float:
        """Get total speed magnitude."""
        return math.sqrt(self.horizontal_speed**2 + self.vertical_speed**2)
    
    def can_accept_task(self) -> bool:
        """Check if UAV can accept new tasks based on energy."""
        return self.energy_mode != EnergyMode.CRITICAL
    
    def compute_flight_duration(self, destination: Location) -> float:
        """Calculate flight time to a destination.
        
        Args:
            destination: Target location.
            
        Returns:
            Flight duration in seconds.
        """
        distance = Location.euclidean_distance_2d(self.location, destination)
        speed = 2.5  # Default speed in m/s
        return distance / speed
    
    def consume_energy(self, amount: float) -> None:
        """Consume battery energy.
        
        Args:
            amount: Energy to consume (reduces battery %).
        """
        self.battery_level = max(0.0, self.battery_level - amount)
        self.energy_tracker.record(amount, "total")
        
        logger.debug(
            f"UAV {self.server_id}: Battery {self.battery_level:.1f}% "
            f"({self.energy_mode.name})"
        )
    
    def consume_flight_energy(
        self,
        distance: float,
        velocity: float = 2.5
    ) -> float:
        """Consume energy for flight.
        
        Args:
            distance: Flight distance in meters.
            velocity: Flight velocity in m/s.
            
        Returns:
            Energy consumed.
        """
        energy = self.energy_model.compute_flight_energy(distance, velocity)
        # Convert to battery percentage (simplified)
        battery_drain = energy / 100.0  # Assuming 100J = 1%
        self.consume_energy(battery_drain)
        self.energy_tracker.record(energy, "flight")
        
        logger.info(
            f"UAV {self.server_id}: Flight {distance:.1f}m consumed {energy:.2f}J"
        )
        return energy
    
    def consume_hover_energy(self, duration: float) -> float:
        """Consume energy for hovering.
        
        Args:
            duration: Hover duration in seconds.
            
        Returns:
            Energy consumed.
        """
        energy = self.energy_model.compute_hover_energy(duration)
        battery_drain = energy / 100.0
        self.consume_energy(battery_drain)
        self.energy_tracker.record(energy, "hover")
        
        logger.debug(f"UAV {self.server_id}: Hover {duration:.1f}s consumed {energy:.2f}J")
        return energy
    
    def consume_communication_energy(self, duration: float) -> float:
        """Consume energy for communication.
        
        Args:
            duration: Communication duration in seconds.
            
        Returns:
            Energy consumed.
        """
        energy = self.energy_model.compute_communication_energy(duration)
        battery_drain = energy / 100.0
        self.consume_energy(battery_drain)
        self.energy_tracker.record(energy, "communication")
        
        return energy
    
    def recharge(self, amount: float = 100.0) -> None:
        """Recharge battery.
        
        Args:
            amount: Amount to recharge (default: full charge).
        """
        old_level = self.battery_level
        self.battery_level = min(100.0, self.battery_level + amount)
        
        logger.info(
            f"UAV {self.server_id}: Recharged {old_level:.1f}% -> {self.battery_level:.1f}%"
        )
    
    def get_status(self) -> dict:
        """Get UAV status summary."""
        return {
            "id": self.server_id,
            "location": str(self.location),
            "battery": self.battery_level,
            "energy_mode": self.energy_mode.name,
            "is_flying": self.is_flying,
            "utilization": self.get_utilization(1000),
            "total_energy": self.get_energy_consumption()
        }
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all UAVs."""
        cls._id_counter = 0
        cls._all_uavs.clear()
        logger.debug("UAV registry reset")
    
    @classmethod
    def get_all(cls) -> List[UAV]:
        """Get all UAVs."""
        return cls._all_uavs.copy()


@dataclass
class CloudServer(Server):
    """Cloud server with high capacity but higher latency.
    
    Cloud servers have virtually unlimited capacity but incur
    higher network latency due to WAN access.
    
    Attributes:
        All inherited from Server.
    """
    
    def __post_init__(self) -> None:
        """Initialize cloud server."""
        object.__setattr__(self, 'server_id', 0)  # Single cloud server
        logger.info(f"CloudServer created with capacity={self.capacity}")
    
    def get_processing_delay(self, task: Task) -> float:
        """Get processing delay including WAN latency.
        
        Cloud has no queueing delay but higher network delay.
        
        Args:
            task: Task to process.
            
        Returns:
            Processing delay in seconds.
        """
        app = task.app
        mu = self.capacity / app.cpu_cycle
        processing_time = 1 / mu
        
        task.processing_time = processing_time
        task.waiting_time_in_queue = 0  # No queueing for cloud
        
        # Track energy (cloud energy is shared/reduced)
        energy = self.energy_model.compute_computation_energy(processing_time) * 0.1
        self.energy_tracker.record(energy, "computation")
        
        return processing_time


# Alias for backward compatibility
def get_all_edge_servers() -> List[EdgeServer]:
    """Get all edge servers."""
    return EdgeServer.get_all()


def get_all_uavs() -> List[UAV]:
    """Get all UAVs."""
    return UAV.get_all()
