"""Core Simulation Engine.

This module provides the main Simulation class that orchestrates
the discrete event simulation for air computing environments.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from aircompsim.config.settings import SimulationConfig
from aircompsim.core.event import Event, EventQueue, EventType
from aircompsim.entities.location import Location, SimulationBoundary
from aircompsim.entities.server import UAV, CloudServer, EdgeServer
from aircompsim.entities.task import Application, ApplicationType, Task
from aircompsim.entities.user import User

if TYPE_CHECKING:
    from aircompsim.drl.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Results from a simulation run.

    Attributes:
        total_tasks: Total tasks generated.
        successful_tasks: Tasks completed within deadline.
        failed_tasks: Tasks that missed deadline.
        avg_latency: Average task latency.
        avg_qos: Average QoS score.
        total_energy: Total energy consumed.
        simulation_time: Actual simulation time.
        uav_stats: Per-UAV statistics.
        edge_stats: Per-edge server statistics.
    """

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_latency: float = 0.0
    avg_qos: float = 0.0
    total_energy: float = 0.0
    simulation_time: float = 0.0
    uav_stats: List[Dict[str, Any]] = field(default_factory=list)
    edge_stats: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Get task success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks


class Simulation:
    """Discrete event simulation engine for air computing.

    Manages the event queue, entity lifecycles, and simulation execution.

    Attributes:
        config: Simulation configuration.
        boundary: Simulation spatial boundary.
        event_queue: Priority queue for events.
        simulation_time: Current simulation time.

    Example:
        >>> config = SimulationConfig(time_limit=1000, user_count=20)
        >>> sim = Simulation(config)
        >>> sim.initialize()
        >>> results = sim.run()
        >>> print(f"Success rate: {results.success_rate:.2%}")
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        """Initialize simulation.

        Args:
            config: Simulation configuration. Uses defaults if not provided.
        """
        self.config = config or SimulationConfig()
        self.boundary = SimulationBoundary(
            max_x=self.config.boundary.max_x,
            max_y=self.config.boundary.max_y,
            max_z=self.config.boundary.max_z,
        )
        self.event_queue = EventQueue()
        self.simulation_time: float = 0.0

        # Entity collections
        self.cloud_server: Optional[CloudServer] = None

        # DRL agent (optional)
        self.agent: Optional[BaseAgent] = None
        self.is_drl_training: bool = False

        # Metrics
        self._task_count: int = 0
        self._successful_tasks: int = 0
        self._total_latency: float = 0.0
        self._total_qos: float = 0.0

        # Event handlers
        self._event_handlers: Dict[EventType, Callable] = {}
        self._setup_event_handlers()

        # Workload RNG
        self.workload_rng = np.random.RandomState()

        logger.info(
            f"Simulation initialized: time_limit={self.config.time_limit}, "
            f"users={self.config.user_count}"
        )

    def _setup_event_handlers(self) -> None:
        """Set up default event handlers."""
        self._event_handlers = {
            EventType.OFFLOAD: self._handle_offload,
            EventType.PROCESS: self._handle_process,
            EventType.RETURNED: self._handle_return,
            EventType.UAV_MOVE: self._handle_uav_move,
            EventType.UAV_STOP: self._handle_uav_stop,
            EventType.USER_MOVE: self._handle_user_move,
            EventType.USER_STOP: self._handle_user_stop,
            EventType.STATE: self._handle_state_update,
        }

    def initialize(self) -> None:
        """Initialize simulation entities.

        Creates users, servers, UAVs, and schedules initial events.
        """
        # Reset registries
        self._reset_entities()

        # Create cloud server
        cloud_location = Location(x=self.boundary.max_x, y=self.boundary.max_y, z=0)
        self.cloud_server = CloudServer(
            capacity=self.config.cloud.capacity,
            location=cloud_location,
            radius=self.boundary.max_x,
            power_consumption=100,
        )

        # Initialize workload RNG
        workload_seed = (
            self.config.workload_seed
            if self.config.workload_seed is not None
            else self.config.seed
        )
        if workload_seed is not None:
            self.workload_rng.seed(workload_seed)

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            print(f"DEBUG: Seeded with {self.config.seed}. Random state: {random.getstate()[1][0]}")

        # ... (skip to _create_users)
        
        # Create edge servers
        self._create_edge_servers()

        # Create UAVs
        self._create_uavs()

        # Create users with applications
        self._create_users()

        # Schedule DRL state events if enabled
        if self.config.drl.enabled:
            self._schedule_state_events()

        logger.info(
            f"Simulation initialized: {len(EdgeServer.get_all())} edges, "
            f"{len(UAV.get_all())} UAVs, {len(User.get_all())} users"
        )

    def _reset_entities(self) -> None:
        """Reset all entity registries."""
        EdgeServer.reset_all()
        UAV.reset_all()
        User.reset_all()
        Application.reset_all()
        Task.reset_all()
        Event.reset_counter()
        ApplicationType.clear_registry()

        self.event_queue.clear()
        self.simulation_time = 0.0
        self._task_count = 0
        self._successful_tasks = 0
        self._total_latency = 0.0
        self._total_qos = 0.0

    def _create_edge_servers(self) -> None:
        """Create edge servers based on configuration."""
        count = self.config.edge.count

        locations = self.config.edge.locations or self._generate_grid_locations(count)

        for x, y in locations[:count]:
            EdgeServer(
                capacity=self.config.edge.capacity,
                location=Location(x=x, y=y, z=0),
                radius=self.config.edge.radius,
                power_consumption=self.config.edge.power,
            )

    def _create_uavs(self) -> None:
        """Create UAVs based on configuration."""
        for _ in range(self.config.uav.count):
            UAV(
                capacity=self.config.uav.capacity,
                location=Location(
                    x=np.random.uniform(0, self.boundary.max_x),
                    y=np.random.uniform(0, self.boundary.max_y),
                    z=self.config.uav.altitude,
                ),
                radius=self.config.uav.radius,
                power_consumption=self.config.uav.power,
                battery_level=self.config.uav.initial_battery,
            )

    def _create_users(self) -> None:
        """Create users with applications."""
        # Create default app types
        app_types = [
            ApplicationType(
                name=app.name,
                cpu_cycle=app.cpu_cycle,
                worst_delay=app.worst_delay,
                best_delay=app.best_delay,
                interarrival_time=app.interarrival_time,
            )
            for app in self.config.applications
        ]

        for _ in range(self.config.user_count):
            user = User(location=self.boundary.random_location(altitude=0))

            # Assign random application
            if app_types:
                import random

                idx = self.workload_rng.randint(0, len(app_types))
                app_type = app_types[idx]
                app = Application(app_type=app_type, start_time=0)
                user.add_application(app)

    def _generate_grid_locations(self, count: int) -> List[tuple[float, float]]:
        """Generate evenly distributed locations."""
        side = int(np.ceil(np.sqrt(count)))
        step_x = self.boundary.max_x / (side + 1)
        step_y = self.boundary.max_y / (side + 1)

        locations = []
        for i in range(side):
            for j in range(side):
                locations.append((step_x * (i + 1), step_y * (j + 1)))

        return locations

    def _schedule_state_events(self) -> None:
        """Schedule DRL state update events."""
        interval = self.config.drl.state_interval
        for t in range(1, int(self.config.time_limit), interval):
            event = Event.create_state_event(time=float(t))
            self.event_queue.push(event)

    def run(self) -> SimulationResults:
        """Run the simulation.

        Returns:
            SimulationResults with metrics.
        """
        logger.info(f"Starting simulation (time_limit={self.config.time_limit})")

        while self.simulation_time < self.config.time_limit:
            # Generate tasks from users
            self._generate_user_tasks()

            # Process events
            if not self.event_queue:
                self.simulation_time += 1
                continue

            event = self.event_queue.pop()
            if event is None:
                continue

            self.simulation_time = event.scheduled_time

            if self.simulation_time >= self.config.time_limit:
                break

            self._process_event(event)

        # Final check for tasks up to time limit
        self.simulation_time = self.config.time_limit
        self._generate_user_tasks()

        results = self._collect_results()
        logger.info(
            f"Simulation complete: {results.total_tasks} tasks, "
            f"{results.success_rate:.2%} success rate"
        )

        return results

    def _generate_user_tasks(self) -> None:
        """Generate tasks from user applications."""
        for user in User.get_all():
            for app in user.applications:
                while app.is_task_valid(self.simulation_time):
                    task = app.generate_task(user, rng=self.workload_rng)
                    if task.creation_time < self.config.time_limit:
                        event = Event.create_offload_event(
                            task=task, time=task.creation_time, location=user.location
                        )
                        self.event_queue.push(event)
                    else:
                        break

    def _process_event(self, event: Event) -> None:
        """Process a single event.

        Args:
            event: Event to process.
        """
        handler = self._event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            logger.warning(f"No handler for event type: {event.event_type}")

    def _handle_offload(self, event: Event) -> None:
        """Handle task offload event."""
        task = event.task
        if task is None:
            return

        # Find best server
        server = self._select_server(task, event.location)

        if server is None:
            logger.warning(f"No server available for task {task.task_id}")
            return

        # Calculate processing delay
        processing_delay = server.get_processing_delay(task)
        completion_time = self.simulation_time + processing_delay

        # Schedule completion event
        completion_event = Event(
            scheduled_time=completion_time, event_type=EventType.RETURNED, task=task
        )
        self.event_queue.push(completion_event)

        task.processed_server = server

    def _handle_process(self, event: Event) -> None:
        """Handle task processing event."""
        pass  # Processing is handled inline

    def _handle_return(self, event: Event) -> None:
        """Handle task return/completion event."""
        task = event.task
        if task is None:
            return

        if task.processed_server:
            task.complete(self.simulation_time, task.processed_server)
        else:
            logger.error(f"Task {task.task_id} returned without processed_server")
            return

        self._task_count += 1
        self._total_latency += task.latency
        self._total_qos += task.qos

        if task.is_success:
            self._successful_tasks += 1

    def _handle_uav_move(self, event: Event) -> None:
        """Handle UAV movement event."""
        pass  # Implemented by subclass or extension

    def _handle_uav_stop(self, event: Event) -> None:
        """Handle UAV stop event."""
        pass

    def _handle_user_move(self, event: Event) -> None:
        """Handle user movement event."""
        user = event.user
        if user and event.location:
            user.move_to(event.location)

    def _handle_user_stop(self, event: Event) -> None:
        """Handle user stop event."""
        user = event.user
        if user:
            user.stop_moving()

    def _handle_state_update(self, _event: Event) -> None:
        """Handle DRL state update event."""
        if self.agent and self.is_drl_training:
            _state = self._get_state()
            # DRL training logic would go here

    def _select_server(
        self, _task: Task, location: Optional[Location]
    ) -> Optional[EdgeServer | UAV]:
        """Select best server for task.

        Args:
            task: Task to schedule.
            location: Task origin location.

        Returns:
            Selected server or None.
        """
        if location is None:
            return None

        # Find covering servers
        candidates: List[Union[EdgeServer, UAV]] = []

        for edge in EdgeServer.get_all():
            if edge.is_in_coverage(location):
                candidates.append(edge)

        for uav in UAV.get_all():
            if uav.is_in_coverage(location) and uav.can_accept_task():
                candidates.append(uav)

        if not candidates:
            return None

        # Simple selection: least utilized
        return min(candidates, key=lambda s: s.get_utilization(self.simulation_time))

    def _get_state(self) -> np.ndarray:
        """Get current state for DRL.

        Returns:
            State vector.
        """
        uav_positions = []
        for uav in UAV.get_all():
            uav_positions.extend(
                [
                    uav.location.x / self.boundary.max_x,
                    uav.location.y / self.boundary.max_y,
                    uav.battery_level / 100.0,
                ]
            )

        return np.array(uav_positions, dtype=np.float32)  # type: ignore

    def _collect_results(self) -> SimulationResults:
        """Collect simulation results."""
        # UAV stats
        uav_stats = [uav.get_status() for uav in UAV.get_all()]

        # Edge stats
        edge_stats = [
            {
                "id": edge.server_id,
                "utilization": edge.get_utilization(self.simulation_time),
                "energy": edge.get_energy_consumption(),
            }
            for edge in EdgeServer.get_all()
        ]

        # Total energy
        total_energy = sum(uav.get_energy_consumption() for uav in UAV.get_all()) + sum(
            edge.get_energy_consumption() for edge in EdgeServer.get_all()
        )

        total_tasks = len(Task.get_all())

        return SimulationResults(
            total_tasks=total_tasks,
            successful_tasks=self._successful_tasks,
            failed_tasks=total_tasks - self._successful_tasks,
            avg_latency=self._total_latency / max(1, self._task_count),
            avg_qos=self._total_qos / max(1, self._task_count),
            total_energy=total_energy,
            simulation_time=self.simulation_time,
            uav_stats=uav_stats,
            edge_stats=edge_stats,
        )

    def set_agent(self, agent: BaseAgent) -> None:
        """Set DRL agent for training.

        Args:
            agent: DRL agent.
        """
        self.agent = agent
        self.is_drl_training = True
