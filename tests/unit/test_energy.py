"""Unit tests for Energy models."""

import pytest

from aircompsim.energy.models import (
    EnergyModel,
    EnergyMode,
    EnergyCoefficients,
    EnergyTracker,
)


class TestEnergyMode:
    """Tests for EnergyMode enum."""
    
    def test_energy_modes_exist(self):
        """Test all energy modes are defined."""
        assert EnergyMode.HIGH is not None
        assert EnergyMode.MEDIUM is not None
        assert EnergyMode.LOW is not None
        assert EnergyMode.CRITICAL is not None


class TestEnergyCoefficients:
    """Tests for EnergyCoefficients dataclass."""
    
    def test_default_coefficients(self):
        """Test default coefficient values."""
        coeffs = EnergyCoefficients()
        
        assert coeffs.alpha == 0.05
        assert coeffs.beta == 3.0
        assert coeffs.gamma == 20.0
        assert coeffs.delta == 10.0
    
    def test_custom_coefficients(self):
        """Test custom coefficient values."""
        coeffs = EnergyCoefficients(
            alpha=0.1,
            beta=5.0,
            gamma=30.0,
            delta=15.0
        )
        
        assert coeffs.alpha == 0.1
        assert coeffs.beta == 5.0


class TestEnergyModel:
    """Tests for EnergyModel class."""
    
    def test_default_initialization(self):
        """Test default model initialization."""
        model = EnergyModel()
        
        assert model.coefficients is not None
        assert model.use_physics_model is False
    
    def test_flight_energy_calculation(self):
        """Test basic flight energy calculation."""
        model = EnergyModel()
        
        # E = α × d × v² = 0.05 × 100 × 10² = 500
        energy = model.compute_flight_energy(distance=100, velocity=10)
        
        assert energy == pytest.approx(500.0)
    
    def test_flight_energy_zero_distance(self):
        """Test flight energy with zero distance."""
        model = EnergyModel()
        
        energy = model.compute_flight_energy(distance=0, velocity=10)
        
        assert energy == 0.0
    
    def test_flight_energy_negative_distance_raises(self):
        """Test flight energy raises for negative distance."""
        model = EnergyModel()
        
        with pytest.raises(ValueError):
            model.compute_flight_energy(distance=-10, velocity=10)
    
    def test_flight_energy_negative_velocity_raises(self):
        """Test flight energy raises for negative velocity."""
        model = EnergyModel()
        
        with pytest.raises(ValueError):
            model.compute_flight_energy(distance=100, velocity=-5)
    
    def test_hover_energy_calculation(self):
        """Test hover energy calculation."""
        model = EnergyModel()
        
        # E = β × t = 3.0 × 10 = 30
        energy = model.compute_hover_energy(duration=10)
        
        assert energy == pytest.approx(30.0)
    
    def test_hover_energy_negative_duration_raises(self):
        """Test hover energy raises for negative duration."""
        model = EnergyModel()
        
        with pytest.raises(ValueError):
            model.compute_hover_energy(duration=-5)
    
    def test_computation_energy_calculation(self):
        """Test computation energy calculation."""
        model = EnergyModel()
        
        # E = γ × t = 20.0 × 5 = 100
        energy = model.compute_computation_energy(duration=5)
        
        assert energy == pytest.approx(100.0)
    
    def test_communication_energy_calculation(self):
        """Test communication energy calculation."""
        model = EnergyModel()
        
        # E = δ × t = 10.0 × 3 = 30
        energy = model.compute_communication_energy(duration=3)
        
        assert energy == pytest.approx(30.0)
    
    def test_total_energy_calculation(self):
        """Test total energy calculation."""
        model = EnergyModel()
        
        total = model.compute_total_energy(
            distance=100,
            velocity=10,
            hover_time=10,
            computation_time=5,
            communication_time=3
        )
        
        expected = 500 + 30 + 100 + 30  # flight + hover + comp + comm
        assert total == pytest.approx(expected)
    
    def test_energy_summary(self):
        """Test energy summary breakdown."""
        model = EnergyModel()
        
        summary = model.get_energy_summary(
            distance=100,
            velocity=10,
            hover_time=10,
            computation_time=5,
            communication_time=3
        )
        
        assert summary["flight_energy"] == pytest.approx(500.0)
        assert summary["hover_energy"] == pytest.approx(30.0)
        assert summary["computation_energy"] == pytest.approx(100.0)
        assert summary["communication_energy"] == pytest.approx(30.0)
        assert summary["total_energy"] == pytest.approx(660.0)
    
    def test_determine_energy_mode_high(self):
        """Test energy mode determination for high battery."""
        mode = EnergyModel.determine_energy_mode(battery_level=80)
        assert mode == EnergyMode.HIGH
    
    def test_determine_energy_mode_medium(self):
        """Test energy mode determination for medium battery."""
        mode = EnergyModel.determine_energy_mode(battery_level=50)
        assert mode == EnergyMode.MEDIUM
    
    def test_determine_energy_mode_low(self):
        """Test energy mode determination for low battery."""
        mode = EnergyModel.determine_energy_mode(battery_level=20)
        assert mode == EnergyMode.LOW
    
    def test_determine_energy_mode_critical(self):
        """Test energy mode determination for critical battery."""
        mode = EnergyModel.determine_energy_mode(battery_level=5)
        assert mode == EnergyMode.CRITICAL


class TestEnergyTracker:
    """Tests for EnergyTracker class."""
    
    def test_initial_state(self):
        """Test initial tracker state."""
        tracker = EnergyTracker()
        
        assert tracker.total_consumed == 0.0
        assert tracker.flight_energy == 0.0
        assert tracker.hover_energy == 0.0
    
    def test_record_flight_energy(self):
        """Test recording flight energy."""
        tracker = EnergyTracker()
        
        tracker.record(100.0, "flight", timestamp=10.0)
        
        assert tracker.total_consumed == 100.0
        assert tracker.flight_energy == 100.0
    
    def test_record_multiple_types(self):
        """Test recording multiple energy types."""
        tracker = EnergyTracker()
        
        tracker.record(100.0, "flight")
        tracker.record(50.0, "hover")
        tracker.record(30.0, "computation")
        
        assert tracker.total_consumed == 180.0
        assert tracker.flight_energy == 100.0
        assert tracker.hover_energy == 50.0
        assert tracker.computation_energy == 30.0
    
    def test_get_breakdown(self):
        """Test getting energy breakdown."""
        tracker = EnergyTracker()
        tracker.record(100.0, "flight")
        tracker.record(50.0, "hover")
        
        breakdown = tracker.get_breakdown()
        
        assert breakdown["total"] == 150.0
        assert breakdown["flight"] == 100.0
        assert breakdown["hover"] == 50.0
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = EnergyTracker()
        tracker.record(100.0, "flight")
        tracker.record(50.0, "hover")
        
        tracker.reset()
        
        assert tracker.total_consumed == 0.0
        assert tracker.flight_energy == 0.0
