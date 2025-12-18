"""Unit tests for ChargingStation and ChargingStationRegistry."""

from unittest.mock import MagicMock

import pytest

from aircompsim.energy.charging import ChargingStation, ChargingStationRegistry
from aircompsim.entities.location import Location


class TestChargingStation:
    """Tests for ChargingStation class."""

    @pytest.fixture(autouse=True)
    def reset_counter(self):
        """Reset station ID counter before each test."""
        ChargingStation.reset_counter()
        yield
        ChargingStation.reset_counter()

    @pytest.fixture
    def station(self):
        """Create a charging station for testing."""
        return ChargingStation(
            location=Location(100, 100, 0),
            capacity=2,
            charging_rate=1.0,
        )

    @pytest.fixture
    def mock_uav(self):
        """Create a mock UAV for testing."""
        uav = MagicMock()
        uav.id = 1
        return uav

    def test_station_creation(self, station):
        """Test charging station creation."""
        assert station.station_id > 0
        assert station.capacity == 2
        assert station.charging_rate == 1.0

    def test_unique_station_ids(self):
        """Test that stations get unique IDs."""
        s1 = ChargingStation(location=Location(0, 0, 0))
        s2 = ChargingStation(location=Location(10, 10, 0))
        s3 = ChargingStation(location=Location(20, 20, 0))

        assert s1.station_id != s2.station_id
        assert s2.station_id != s3.station_id

    def test_available_slots(self, station):
        """Test available slots calculation."""
        assert station.available_slots == 2

    def test_is_available(self, station):
        """Test availability check."""
        assert station.is_available is True

    def test_utilization_empty(self, station):
        """Test utilization when empty."""
        assert station.utilization == 0.0

    def test_start_charging_success(self, station, mock_uav):
        """Test starting charging successfully."""
        result = station.start_charging(mock_uav)

        assert result is True
        assert station.available_slots == 1
        assert station.is_charging(mock_uav) is True

    def test_start_charging_no_slots(self, station):
        """Test starting charging when no slots available."""
        # Fill all slots
        uav1 = MagicMock()
        uav1.id = 1
        uav2 = MagicMock()
        uav2.id = 2
        uav3 = MagicMock()
        uav3.id = 3

        station.start_charging(uav1)
        station.start_charging(uav2)

        result = station.start_charging(uav3)

        assert result is False
        assert station.available_slots == 0

    def test_start_charging_already_charging(self, station, mock_uav):
        """Test starting charging for UAV already charging."""
        station.start_charging(mock_uav)

        result = station.start_charging(mock_uav)

        assert result is False

    def test_stop_charging_success(self, station, mock_uav):
        """Test stopping charging successfully."""
        station.start_charging(mock_uav)

        result = station.stop_charging(mock_uav)

        assert result is True
        assert station.available_slots == 2
        assert station.is_charging(mock_uav) is False

    def test_stop_charging_not_charging(self, station, mock_uav):
        """Test stopping charging for UAV not charging."""
        result = station.stop_charging(mock_uav)

        assert result is False

    def test_is_charging(self, station, mock_uav):
        """Test is_charging check."""
        assert station.is_charging(mock_uav) is False

        station.start_charging(mock_uav)
        assert station.is_charging(mock_uav) is True

    def test_get_status(self, station, mock_uav):
        """Test getting station status."""
        station.start_charging(mock_uav)

        status = station.get_status()

        assert status["station_id"] == station.station_id
        assert status["capacity"] == 2
        assert status["occupied_slots"] == 1
        assert status["available_slots"] == 1
        assert mock_uav.id in status["charging_uavs"]

    def test_distance_to(self, station):
        """Test distance calculation."""
        target = Location(200, 100, 0)

        distance = station.distance_to(target)

        assert distance == pytest.approx(100.0)

    def test_utilization_calculation(self, station, mock_uav):
        """Test utilization calculation."""
        assert station.utilization == 0.0

        station.start_charging(mock_uav)
        assert station.utilization == 0.5

        uav2 = MagicMock()
        uav2.id = 2
        station.start_charging(uav2)
        assert station.utilization == 1.0


class TestChargingStationRegistry:
    """Tests for ChargingStationRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_counter(self):
        """Reset station ID counter before each test."""
        ChargingStation.reset_counter()
        yield
        ChargingStation.reset_counter()

    @pytest.fixture
    def registry(self):
        """Create a registry for testing."""
        return ChargingStationRegistry()

    @pytest.fixture
    def stations(self):
        """Create multiple stations for testing."""
        return [
            ChargingStation(location=Location(0, 0, 0), capacity=2),
            ChargingStation(location=Location(100, 0, 0), capacity=2),
            ChargingStation(location=Location(200, 0, 0), capacity=2),
        ]

    def test_add_station(self, registry, stations):
        """Test adding stations to registry."""
        registry.add_station(stations[0])

        assert registry.station_count == 1

    def test_remove_station(self, registry, stations):
        """Test removing station from registry."""
        registry.add_station(stations[0])
        station_id = stations[0].station_id

        result = registry.remove_station(station_id)

        assert result is True
        assert registry.station_count == 0

    def test_remove_nonexistent_station(self, registry):
        """Test removing nonexistent station."""
        result = registry.remove_station(99999)

        assert result is False

    def test_get_station(self, registry, stations):
        """Test getting station by ID."""
        registry.add_station(stations[0])
        station_id = stations[0].station_id

        found = registry.get_station(station_id)

        assert found == stations[0]

    def test_get_nonexistent_station(self, registry):
        """Test getting nonexistent station."""
        found = registry.get_station(99999)

        assert found is None

    def test_find_nearest_available(self, registry, stations):
        """Test finding nearest available station."""
        for s in stations:
            registry.add_station(s)

        user_location = Location(50, 0, 0)
        nearest = registry.find_nearest_available(user_location)

        # Station at (0,0) is closest to (50,0)
        assert nearest == stations[0]

    def test_find_nearest_available_none_available(self, registry, stations):
        """Test finding nearest when none available."""
        station = stations[0]
        station._occupied_slots = station.capacity  # Fill all slots
        registry.add_station(station)

        nearest = registry.find_nearest_available(Location(50, 0, 0))

        assert nearest is None

    def test_find_all_in_range(self, registry, stations):
        """Test finding all stations in range."""
        for s in stations:
            registry.add_station(s)

        in_range = registry.find_all_in_range(Location(0, 0, 0), max_distance=150)

        assert len(in_range) == 2
        assert stations[0] in in_range
        assert stations[1] in in_range
        assert stations[2] not in in_range

    def test_station_count(self, registry, stations):
        """Test station count property."""
        assert registry.station_count == 0

        for s in stations:
            registry.add_station(s)

        assert registry.station_count == 3

    def test_total_available_slots(self, registry, stations):
        """Test total available slots calculation."""
        for s in stations:
            registry.add_station(s)

        assert registry.total_available_slots == 6  # 3 stations * 2 slots

    def test_get_all_statuses(self, registry, stations):
        """Test getting all station statuses."""
        for s in stations:
            registry.add_station(s)

        statuses = registry.get_all_statuses()

        assert len(statuses) == 3
        assert all("station_id" in status for status in statuses)

    def test_clear(self, registry, stations):
        """Test clearing registry."""
        for s in stations:
            registry.add_station(s)

        registry.clear()

        assert registry.station_count == 0
