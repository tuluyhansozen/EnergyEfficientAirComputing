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
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from aircompsim.core.simulation import Simulation, SimulationResults
from aircompsim.core.event import Event, EventType, EventQueue

# Entity imports
from aircompsim.entities.location import Location, SimulationBoundary
from aircompsim.entities.server import EdgeServer, UAV, CloudServer
from aircompsim.entities.user import User
from aircompsim.entities.task import Task, Application, ApplicationType

# Energy imports
from aircompsim.energy.models import EnergyModel, EnergyMode, EnergyTracker

# Config imports
from aircompsim.config.settings import SimulationConfig

__all__ = [
    # Core
    "Simulation",
    "SimulationResults",
    "Event",
    "EventType",
    "EventQueue",
    # Entities
    "Location",
    "SimulationBoundary",
    "EdgeServer",
    "UAV",
    "CloudServer",
    "User",
    "Task",
    "Application",
    "ApplicationType",
    # Energy
    "EnergyModel",
    "EnergyMode",
    "EnergyTracker",
    # Config
    "SimulationConfig",
    # Metadata
    "__version__",
]
