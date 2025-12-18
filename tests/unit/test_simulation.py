"""Unit tests for Simulation core module."""

import pytest

from aircompsim.config.settings import SimulationConfig
from aircompsim.core.simulation import Simulation, SimulationResults
from aircompsim.entities.server import UAV, CloudServer, EdgeServer
from aircompsim.entities.user import User


class TestSimulationResults:
    """Tests for SimulationResults dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        results = SimulationResults(
            total_tasks=100,
            successful_tasks=75,
            failed_tasks=25,
        )

        assert results.success_rate == 0.75

    def test_success_rate_zero_tasks(self):
        """Test success rate with zero tasks."""
        results = SimulationResults(
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0,
        )

        assert results.success_rate == 0.0

    def test_default_values(self):
        """Test default values."""
        results = SimulationResults()

        assert results.total_tasks == 0
        assert results.successful_tasks == 0
        assert results.avg_latency == 0.0
        assert results.total_energy == 0.0


class TestSimulation:
    """Tests for Simulation class."""

    @pytest.fixture(autouse=True)
    def reset_entities(self):
        """Reset all entity registries before each test."""
        User.reset_all()
        EdgeServer.reset_all()
        UAV.reset_all()
        CloudServer._instance = None
        yield
        User.reset_all()
        EdgeServer.reset_all()
        UAV.reset_all()
        CloudServer._instance = None

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = SimulationConfig()
        config.time_limit = 100
        config.user_count = 5
        config.uav.count = 2
        config.edge.count = 2
        return config

    @pytest.fixture
    def simulation(self, config):
        """Create a simulation instance."""
        return Simulation(config)

    def test_simulation_creation_default_config(self):
        """Test simulation creation with default config."""
        sim = Simulation()

        assert sim.config is not None
        assert sim.simulation_time == 0.0

    def test_simulation_creation_custom_config(self, config):
        """Test simulation creation with custom config."""
        sim = Simulation(config)

        assert sim.config.time_limit == 100
        assert sim.config.user_count == 5

    def test_initialize_creates_entities(self, simulation):
        """Test that initialize creates entities."""
        simulation.initialize()

        assert len(User.get_all()) == 5
        assert len(EdgeServer.get_all()) == 2
        assert len(UAV.get_all()) == 2

    def test_run_returns_results(self, simulation):
        """Test that run returns SimulationResults."""
        simulation.initialize()

        results = simulation.run()

        assert isinstance(results, SimulationResults)

    def test_run_updates_simulation_time(self, simulation):
        """Test that run updates simulation time."""
        simulation.initialize()

        simulation.run()

        assert simulation.simulation_time > 0

    def test_boundary_property(self, simulation):
        """Test simulation boundary property."""
        assert simulation.boundary is not None
        assert simulation.boundary.max_x > 0
        assert simulation.boundary.max_y > 0


class TestSimulationIntegration:
    """Integration tests for Simulation."""

    @pytest.fixture(autouse=True)
    def reset_entities(self):
        """Reset all entity registries."""
        User.reset_all()
        EdgeServer.reset_all()
        UAV.reset_all()
        CloudServer._instance = None
        yield
        User.reset_all()
        EdgeServer.reset_all()
        UAV.reset_all()
        CloudServer._instance = None

    def test_full_simulation_run(self):
        """Test a complete simulation run."""
        config = SimulationConfig()
        config.time_limit = 50
        config.user_count = 3
        config.uav.count = 1
        config.edge.count = 2

        sim = Simulation(config)
        sim.initialize()
        results = sim.run()

        assert results.simulation_time > 0
        assert isinstance(results.avg_latency, float)
        assert isinstance(results.total_energy, float)
