"""Unit tests for Battery models."""

import pytest

from aircompsim.energy.battery import BatteryModel, BatterySpecifications
from aircompsim.energy.models import EnergyMode


class TestBatterySpecifications:
    """Tests for BatterySpecifications dataclass."""

    def test_default_specifications(self):
        """Test default battery specifications."""
        specs = BatterySpecifications()

        assert specs.capacity == 5935.0
        assert specs.voltage == 52.8
        assert specs.max_discharge_rate == 2.0
        assert specs.max_charge_rate == 1.0
        assert specs.max_cycles == 400

    def test_custom_specifications(self):
        """Test custom battery specifications."""
        specs = BatterySpecifications(
            capacity=10000.0,
            voltage=48.0,
            max_discharge_rate=3.0,
            max_charge_rate=2.0,
            max_cycles=500,
        )

        assert specs.capacity == 10000.0
        assert specs.voltage == 48.0
        assert specs.max_cycles == 500


class TestBatteryModel:
    """Tests for BatteryModel class."""

    @pytest.fixture
    def battery(self):
        """Create a default battery for testing."""
        return BatteryModel()

    @pytest.fixture
    def custom_battery(self):
        """Create a custom battery for testing."""
        specs = BatterySpecifications(capacity=1000.0)
        return BatteryModel(specs=specs, state_of_charge=50.0)

    def test_default_initialization(self, battery):
        """Test default battery initialization."""
        assert battery.state_of_charge == 100.0
        assert battery.health == 1.0  # Health is 0-1, not 0-100
        assert battery.cycle_count == 0
        assert battery._temperature == 25.0  # Private attribute

    def test_effective_capacity(self, battery):
        """Test effective capacity calculation."""
        # At 100% health, effective capacity equals nominal
        effective = battery.effective_capacity

        assert effective > 0
        assert effective <= battery.specs.capacity

    def test_available_energy(self, battery):
        """Test available energy calculation."""
        energy = battery.available_energy

        assert energy > 0
        # Should be in Joules (Wh * 3600), health is 0-1
        assert energy == pytest.approx(battery.specs.capacity * 3600 * battery.health, rel=0.1)

    def test_energy_mode_high(self, battery):
        """Test HIGH energy mode at full charge."""
        battery.state_of_charge = 90.0

        assert battery.energy_mode == EnergyMode.HIGH

    def test_energy_mode_medium(self, battery):
        """Test MEDIUM energy mode."""
        battery.state_of_charge = 60.0

        assert battery.energy_mode == EnergyMode.MEDIUM

    def test_energy_mode_low(self, battery):
        """Test LOW energy mode."""
        battery.state_of_charge = 25.0

        assert battery.energy_mode == EnergyMode.LOW

    def test_energy_mode_critical(self, battery):
        """Test CRITICAL energy mode."""
        battery.state_of_charge = 10.0

        assert battery.energy_mode == EnergyMode.CRITICAL

    def test_discharge_successful(self, custom_battery):
        """Test successful battery discharge."""
        initial_soc = custom_battery.state_of_charge
        energy_to_discharge = 100.0  # Small amount

        result = custom_battery.discharge(energy_to_discharge)

        assert result is True
        assert custom_battery.state_of_charge < initial_soc

    def test_discharge_insufficient_energy(self, custom_battery):
        """Test discharge with insufficient energy."""
        # Try to discharge more than available
        huge_energy = 1e12

        result = custom_battery.discharge(huge_energy)

        assert result is False

    def test_charge_successful(self, custom_battery):
        """Test successful battery charging."""
        initial_soc = custom_battery.state_of_charge

        energy_added = custom_battery.charge(duration_seconds=100.0, charge_rate=1.0)

        assert energy_added > 0
        assert custom_battery.state_of_charge > initial_soc

    def test_charge_at_full(self, battery):
        """Test charging when already full."""
        battery.state_of_charge = 100.0

        energy_added = battery.charge(duration_seconds=100.0)

        assert energy_added == 0

    def test_charge_to_full(self, custom_battery):
        """Test charging to full capacity."""
        custom_battery.state_of_charge = 50.0

        time_needed, energy_added = custom_battery.charge_to_full()

        assert time_needed > 0
        assert energy_added > 0
        assert custom_battery.state_of_charge == pytest.approx(100.0)

    def test_can_complete_mission_yes(self, battery):
        """Test mission feasibility check - possible."""
        small_mission_energy = 1000.0  # Small mission

        result = battery.can_complete_mission(small_mission_energy)

        assert result is True

    def test_can_complete_mission_no(self, custom_battery):
        """Test mission feasibility check - not possible."""
        custom_battery.state_of_charge = 5.0
        large_mission_energy = 1e9  # Huge mission

        result = custom_battery.can_complete_mission(large_mission_energy)

        assert result is False

    def test_estimate_remaining_flight_time(self, battery):
        """Test remaining flight time estimation."""
        power_consumption = 400.0  # Watts

        remaining_time = battery.estimate_remaining_flight_time(power_consumption)

        assert remaining_time > 0

    def test_estimate_remaining_flight_time_zero_power(self, battery):
        """Test remaining flight time with zero power consumption."""
        remaining_time = battery.estimate_remaining_flight_time(0.0)

        assert remaining_time == float("inf")

    def test_get_status(self, battery):
        """Test battery status summary."""
        status = battery.get_status()

        assert "state_of_charge" in status
        assert "health" in status
        assert "energy_mode" in status
        assert "cycle_count" in status
        assert "temperature" in status

    def test_set_temperature(self, battery):
        """Test setting battery temperature."""
        battery.set_temperature(35.0)

        assert battery._temperature == 35.0  # Private attribute

    def test_reset_full_charge(self, custom_battery):
        """Test battery reset with full charge."""
        custom_battery.state_of_charge = 20.0
        custom_battery.cycle_count = 10

        custom_battery.reset(full_charge=True)

        assert custom_battery.state_of_charge == 100.0
        assert custom_battery.cycle_count == 0

    def test_reset_preserve_charge(self, custom_battery):
        """Test battery reset preserving charge."""
        custom_battery.state_of_charge = 50.0

        custom_battery.reset(full_charge=False)

        assert custom_battery.state_of_charge == 50.0

    def test_temperature_effect_on_capacity(self, battery):
        """Test that extreme temperatures affect capacity."""
        battery.set_temperature(25.0)
        normal_capacity = battery.effective_capacity

        battery.set_temperature(-10.0)
        cold_capacity = battery.effective_capacity

        battery.set_temperature(45.0)
        hot_capacity = battery.effective_capacity

        # Cold and hot should have lower effective capacity
        assert cold_capacity <= normal_capacity
        assert hot_capacity <= normal_capacity
