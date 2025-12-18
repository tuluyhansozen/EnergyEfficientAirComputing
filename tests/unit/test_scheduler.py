"""Unit tests for EnergyAwareScheduler."""

from unittest.mock import MagicMock

import pytest

from aircompsim.energy.scheduler import (
    EnergyAwareScheduler,
    SchedulingDecision,
    SchedulingStrategy,
)


class TestSchedulingStrategy:
    """Tests for SchedulingStrategy enum."""

    def test_strategies_exist(self):
        """Test that all strategies are defined."""
        assert SchedulingStrategy.ENERGY_FIRST
        assert SchedulingStrategy.LATENCY_FIRST
        assert SchedulingStrategy.BALANCED
        assert SchedulingStrategy.UTILIZATION


class TestSchedulingDecision:
    """Tests for SchedulingDecision dataclass."""

    def test_valid_decision(self):
        """Test valid scheduling decision."""
        mock_server = MagicMock()
        decision = SchedulingDecision(
            server=mock_server,
            estimated_latency=0.5,
            estimated_energy=100.0,
            confidence=0.9,
            reason="Test",
        )

        assert decision.is_valid is True
        assert decision.server == mock_server
        assert decision.estimated_latency == 0.5

    def test_invalid_decision(self):
        """Test invalid scheduling decision (no server)."""
        decision = SchedulingDecision(
            server=None,
            estimated_latency=0.0,
            estimated_energy=0.0,
            reason="No server available",
        )

        assert decision.is_valid is False


class TestEnergyAwareScheduler:
    """Tests for EnergyAwareScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler for testing."""
        return EnergyAwareScheduler(
            strategy=SchedulingStrategy.BALANCED,
            energy_weight=0.5,
            latency_weight=0.5,
        )

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.strategy == SchedulingStrategy.BALANCED
        assert scheduler.energy_weight == 0.5
        assert scheduler.latency_weight == 0.5

    def test_scheduler_energy_first(self):
        """Test energy-first scheduler initialization."""
        sched = EnergyAwareScheduler(strategy=SchedulingStrategy.ENERGY_FIRST)
        assert sched.strategy == SchedulingStrategy.ENERGY_FIRST

    def test_scheduler_latency_first(self):
        """Test latency-first scheduler initialization."""
        sched = EnergyAwareScheduler(strategy=SchedulingStrategy.LATENCY_FIRST)
        assert sched.strategy == SchedulingStrategy.LATENCY_FIRST

    def test_scheduler_utilization(self):
        """Test utilization scheduler initialization."""
        sched = EnergyAwareScheduler(strategy=SchedulingStrategy.UTILIZATION)
        assert sched.strategy == SchedulingStrategy.UTILIZATION

    def test_select_server_empty_list(self, scheduler):
        """Test select_server with empty server list."""
        mock_task = MagicMock()
        decision = scheduler.select_server(mock_task, [], current_time=0.0)

        assert decision.is_valid is False
        assert decision.server is None

    def test_should_accept_task_non_uav(self, scheduler):
        """Test task acceptance for non-UAV server."""
        mock_task = MagicMock()
        mock_edge = MagicMock(spec=[])  # No battery_level attribute

        result = scheduler.should_accept_task(mock_edge, mock_task)

        assert result is True

    def test_custom_weights(self):
        """Test scheduler with custom weights."""
        sched = EnergyAwareScheduler(
            energy_weight=0.8,
            latency_weight=0.2,
            uav_energy_threshold=30.0,
        )
        assert sched.energy_weight == 0.8
        assert sched.latency_weight == 0.2
        assert sched.uav_energy_threshold == 30.0
