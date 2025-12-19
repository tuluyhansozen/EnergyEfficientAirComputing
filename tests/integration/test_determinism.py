"""Integration tests for deterministic behavior."""

import pytest
from aircompsim.config.settings import SimulationConfig
from aircompsim.core.simulation import Simulation


class TestDeterminism:
    """Tests for deterministic simulation behavior."""

    def test_workload_independence_from_infrastructure(self):
        """Test that workload (tasks) is independent of infrastructure (UAVs)."""
        SEED = 42
        WORKLOAD_SEED = 12345
        TIME_LIMIT = 50.0

        # Scenario 1: No UAVs
        config1 = SimulationConfig(
            time_limit=TIME_LIMIT,
            user_count=10,
            seed=SEED,
            workload_seed=WORKLOAD_SEED,
        )
        config1.uav.count = 0
        sim1 = Simulation(config1)
        sim1.initialize()
        results1 = sim1.run()

        # Scenario 2: Many UAVs
        config2 = SimulationConfig(
            time_limit=TIME_LIMIT,
            user_count=10,
            seed=SEED,
            workload_seed=WORKLOAD_SEED,
        )
        config2.uav.count = 10  # Changing infrastructure
        sim2 = Simulation(config2)
        sim2.initialize()
        results2 = sim2.run()

        # Tasks generated should be IDENTICAL
        assert results1.total_tasks == results2.total_tasks
        assert results1.total_tasks > 0

    def test_workload_determinism(self):
        """Test that same workload seed produces same workload."""
        SEED = 42
        TIME_LIMIT = 50.0

        config1 = SimulationConfig(time_limit=TIME_LIMIT, seed=SEED, workload_seed=SEED)
        sim1 = Simulation(config1)
        sim1.initialize()
        r1 = sim1.run()

        config2 = SimulationConfig(time_limit=TIME_LIMIT, seed=SEED, workload_seed=SEED)
        sim2 = Simulation(config2)
        sim2.initialize()
        r2 = sim2.run()

        assert r1.total_tasks == r2.total_tasks
