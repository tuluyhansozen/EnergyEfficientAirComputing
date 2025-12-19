"""Charging station management for UAVs.

This module provides charging station models for UAV battery management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aircompsim.entities.location import Location
    from aircompsim.entities.server import UAV

logger = logging.getLogger(__name__)


@dataclass
class ChargingStation:
    """Fixed-position UAV charging station.

    Manages charging slots and tracks charging UAVs.

    Attributes:
        location: Station location.
        capacity: Maximum simultaneous charging slots.
        charging_rate: Charging power factor (0-1).
        station_id: Unique station identifier.

    Example:
        >>> station = ChargingStation(
        ...     location=Location(100, 100, 0),
        ...     capacity=2,
        ...     charging_rate=0.5
        ... )
        >>> if station.is_available():
        ...     station.start_charging(uav)
    """

    location: Location
    capacity: int = 2
    charging_rate: float = 1.0
    station_id: int = field(default=0)
    _occupied_slots: int = field(default=0, repr=False)
    _charging_uavs: set = field(default_factory=set, repr=False)

    # Class-level ID counter
    _id_counter: int = 0

    def __post_init__(self) -> None:
        """Assign unique ID after initialization."""
        ChargingStation._id_counter += 1
        object.__setattr__(self, "station_id", ChargingStation._id_counter)

    @property
    def available_slots(self) -> int:
        """Get number of available charging slots."""
        return self.capacity - self._occupied_slots

    @property
    def is_available(self) -> bool:
        """Check if station has available slots."""
        return self._occupied_slots < self.capacity

    @property
    def utilization(self) -> float:
        """Get station utilization as fraction."""
        return self._occupied_slots / self.capacity if self.capacity > 0 else 0.0

    def start_charging(self, uav: UAV) -> bool:
        """Start charging a UAV.

        Args:
            uav: UAV to start charging.

        Returns:
            True if charging started, False if no slots available.
        """
        if not self.is_available:
            logger.warning(f"Station {self.station_id}: No slots available for UAV {uav.id}")
            return False

        if uav.id in self._charging_uavs:
            logger.warning(f"Station {self.station_id}: UAV {uav.id} already charging")
            return False

        self._occupied_slots += 1
        self._charging_uavs.add(uav.id)

        logger.info(
            f"Station {self.station_id}: Started charging UAV {uav.id} "
            f"(slots: {self._occupied_slots}/{self.capacity})"
        )
        return True

    def stop_charging(self, uav: UAV) -> bool:
        """Stop charging a UAV.

        Args:
            uav: UAV to stop charging.

        Returns:
            True if charging stopped, False if UAV wasn't charging.
        """
        if uav.id not in self._charging_uavs:
            logger.warning(f"Station {self.station_id}: UAV {uav.id} was not charging")
            return False

        self._occupied_slots = max(0, self._occupied_slots - 1)
        self._charging_uavs.discard(uav.id)

        logger.info(
            f"Station {self.station_id}: Stopped charging UAV {uav.id} "
            f"(slots: {self._occupied_slots}/{self.capacity})"
        )
        return True

    def is_charging(self, uav: UAV) -> bool:
        """Check if a UAV is currently charging at this station."""
        return uav.id in self._charging_uavs

    def get_status(self) -> dict:
        """Get station status summary."""
        return {
            "station_id": self.station_id,
            "location": str(self.location),
            "capacity": self.capacity,
            "occupied_slots": self._occupied_slots,
            "available_slots": self.available_slots,
            "utilization": self.utilization,
            "charging_uavs": list(self._charging_uavs),
        }

    def distance_to(self, location: Location) -> float:
        """Calculate distance from station to a location.

        Args:
            location: Target location.

        Returns:
            Distance in meters.
        """
        from aircompsim.entities.location import Location as Loc

        return Loc.euclidean_distance_2d(self.location, location)

    @classmethod
    def reset_counter(cls) -> None:
        """Reset station ID counter."""
        cls._id_counter = 0


@dataclass
class ChargingStationRegistry:
    """Registry for managing multiple charging stations.

    Provides methods for finding optimal charging stations
    based on distance and availability.
    """

    _stations: list[ChargingStation] = field(default_factory=list)

    def add_station(self, station: ChargingStation) -> None:
        """Add a charging station to the registry."""
        self._stations.append(station)
        logger.debug(f"Added station {station.station_id} to registry")

    def remove_station(self, station_id: int) -> bool:
        """Remove a charging station from the registry."""
        for i, station in enumerate(self._stations):
            if station.station_id == station_id:
                self._stations.pop(i)
                logger.debug(f"Removed station {station_id} from registry")
                return True
        return False

    def get_station(self, station_id: int) -> ChargingStation | None:
        """Get a station by ID."""
        for station in self._stations:
            if station.station_id == station_id:
                return station
        return None

    def find_nearest_available(self, location: Location) -> ChargingStation | None:
        """Find the nearest available charging station.

        Args:
            location: Reference location.

        Returns:
            Nearest available station, or None if none available.
        """
        available = [s for s in self._stations if s.is_available]

        if not available:
            logger.warning("No available charging stations found")
            return None

        nearest = min(available, key=lambda s: s.distance_to(location))
        logger.debug(
            f"Nearest available station: {nearest.station_id} "
            f"at distance {nearest.distance_to(location):.1f}m"
        )
        return nearest

    def find_all_in_range(self, location: Location, max_distance: float) -> list[ChargingStation]:
        """Find all stations within range.

        Args:
            location: Reference location.
            max_distance: Maximum distance in meters.

        Returns:
            List of stations within range, sorted by distance.
        """
        in_range = [s for s in self._stations if s.distance_to(location) <= max_distance]
        return sorted(in_range, key=lambda s: s.distance_to(location))

    @property
    def station_count(self) -> int:
        """Get total number of stations."""
        return len(self._stations)

    @property
    def total_available_slots(self) -> int:
        """Get total available charging slots across all stations."""
        return sum(s.available_slots for s in self._stations)

    def get_all_statuses(self) -> list[dict]:
        """Get status of all stations."""
        return [s.get_status() for s in self._stations]

    def clear(self) -> None:
        """Clear all stations from registry."""
        self._stations.clear()
        ChargingStation.reset_counter()
        logger.debug("Charging station registry cleared")
