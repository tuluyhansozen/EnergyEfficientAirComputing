"""Test configuration for pytest."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_location():
    """Create a sample location for testing."""
    from aircompsim.entities.location import Location

    return Location(x=100.0, y=200.0, z=0.0)


@pytest.fixture
def sample_boundary():
    """Create a sample simulation boundary."""
    from aircompsim.entities.location import SimulationBoundary

    return SimulationBoundary(max_x=400.0, max_y=400.0, max_z=400.0)


@pytest.fixture
def sample_energy_model():
    """Create a sample energy model."""
    from aircompsim.energy.models import EnergyModel

    return EnergyModel()


@pytest.fixture
def sample_config():
    """Create a sample simulation config."""
    from aircompsim.config.settings import SimulationConfig

    return SimulationConfig()


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset all entity registries before each test."""
    yield
    # Cleanup after test
    try:
        from aircompsim.entities.server import EdgeServer, UAV
        from aircompsim.entities.user import User
        from aircompsim.entities.task import Application, Task, ApplicationType
        from aircompsim.core.event import Event

        EdgeServer.reset_all()
        UAV.reset_all()
        User.reset_all()
        Application.reset_all()
        ApplicationType.clear_registry()
        Event.reset_counter()
    except ImportError:
        pass  # Modules not yet created
