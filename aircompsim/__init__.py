"""
AirCompSim - Energy-Efficient Air Computing Discrete Event Simulator.

A modular simulation framework for air computing environments integrating
UAVs, edge servers, and cloud computing with energy-aware task offloading.

Example:
    >>> from aircompsim import Simulation, SimulationConfig
    >>> config = SimulationConfig(time_limit=1000, user_count=20)
    >>> sim = Simulation(config)
    >>> sim.initialize()
    >>> results = sim.run()
    >>> print(f"Success rate: {results.success_rate:.2%}")
"""

__version__ = "1.0.0"

# Core imports
# Config imports
from aircompsim.config.settings import SimulationConfig
from aircompsim.core.event import Event, EventQueue, EventType
from aircompsim.core.simulation import Simulation, SimulationResults

# Energy imports
from aircompsim.energy.models import EnergyMode, EnergyModel, EnergyTracker

# Entity imports
from aircompsim.entities.location import Location, SimulationBoundary
from aircompsim.entities.server import UAV, CloudServer, EdgeServer
from aircompsim.entities.task import Application, ApplicationType, Task
from aircompsim.entities.user import User

__all__ = [  # noqa: RUF022
    # Core
    "Event",
    "EventQueue",
    "EventType",
    "Simulation",
    "SimulationResults",
    # Entities
    "Application",
    "ApplicationType",
    "CloudServer",
    "EdgeServer",
    "Location",
    "SimulationBoundary",
    "Task",
    "UAV",
    "User",
    # Energy
    "EnergyMode",
    "EnergyModel",
    "EnergyTracker",
    # Config
    "SimulationConfig",
    # Metadata
    "__version__",
]
