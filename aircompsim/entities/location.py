"""Location representation with distance calculations.

This module provides the Location class for representing 3D coordinates
in the simulation environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Location:
    """Represents a 3D position in the simulation environment.

    Attributes:
        x: X-coordinate in meters.
        y: Y-coordinate in meters.
        z: Z-coordinate (altitude) in meters.

    Example:
        >>> loc1 = Location(100, 200, 0)
        >>> loc2 = Location(150, 250, 0)
        >>> distance = Location.euclidean_distance_2d(loc1, loc2)
        >>> print(f"Distance: {distance:.2f}m")
        Distance: 70.71m
    """

    x: float
    y: float
    z: float = 0.0

    def __str__(self) -> str:
        """Return string representation of location."""
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def __repr__(self) -> str:
        """Return detailed representation of location."""
        return f"Location(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another location."""
        if not isinstance(other, Location):
            return NotImplemented
        return (
            math.isclose(self.x, other.x)
            and math.isclose(self.y, other.y)
            and math.isclose(self.z, other.z)
        )

    def __hash__(self) -> int:
        """Make Location hashable for use in sets/dicts."""
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))

    @property
    def terrestrial(self) -> Tuple[float, float]:
        """Get 2D terrestrial coordinates (x, y)."""
        return (self.x, self.y)

    @property
    def coordinates(self) -> Tuple[float, float, float]:
        """Get full 3D coordinates (x, y, z)."""
        return (self.x, self.y, self.z)

    def distance_to(self, other: Location, use_3d: bool = False) -> float:
        """Calculate distance to another location.

        Args:
            other: Target location.
            use_3d: If True, calculate 3D Euclidean distance.
                   If False, calculate 2D distance (ignoring altitude).

        Returns:
            Distance in meters.

        Raises:
            TypeError: If other is not a Location instance.
        """
        if not isinstance(other, Location):
            raise TypeError(f"Expected Location, got {type(other).__name__}")

        if use_3d:
            return self.euclidean_distance_3d(self, other)
        return self.euclidean_distance_2d(self, other)

    @staticmethod
    def euclidean_distance_2d(loc1: Location, loc2: Location) -> float:
        """Calculate 2D Euclidean distance between two locations.

        Args:
            loc1: First location.
            loc2: Second location.

        Returns:
            2D distance in meters (ignoring z-coordinate).
        """
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def euclidean_distance_3d(loc1: Location, loc2: Location) -> float:
        """Calculate 3D Euclidean distance between two locations.

        Args:
            loc1: First location.
            loc2: Second location.

        Returns:
            3D distance in meters.
        """
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.z - loc2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @classmethod
    def random_within(
        cls,
        max_x: float,
        max_y: float,
        altitude: float = 0.0,
        min_x: float = 0.0,
        min_y: float = 0.0,
    ) -> Location:
        """Generate a random location within specified bounds.

        Args:
            max_x: Maximum x-coordinate.
            max_y: Maximum y-coordinate.
            altitude: Fixed z-coordinate (default: 0).
            min_x: Minimum x-coordinate (default: 0).
            min_y: Minimum y-coordinate (default: 0).

        Returns:
            Random Location within bounds.

        Raises:
            ValueError: If min coordinates exceed max coordinates.
        """
        if min_x > max_x or min_y > max_y:
            raise ValueError("Minimum coordinates cannot exceed maximum coordinates")

        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        return cls(x=x, y=y, z=altitude)

    def copy(self) -> Location:
        """Create a copy of this location."""
        return Location(x=self.x, y=self.y, z=self.z)

    def move_towards(self, target: Location, distance: float) -> Location:
        """Calculate new location after moving towards target.

        Args:
            target: Target location to move towards.
            distance: Distance to move in meters.

        Returns:
            New location after movement.
        """
        current_distance = self.distance_to(target)

        if current_distance == 0:
            return self.copy()

        # Calculate unit vector towards target
        ratio = min(distance / current_distance, 1.0)
        new_x = self.x + (target.x - self.x) * ratio
        new_y = self.y + (target.y - self.y) * ratio
        new_z = self.z + (target.z - self.z) * ratio

        return Location(x=new_x, y=new_y, z=new_z)


@dataclass
class SimulationBoundary:
    """Defines the boundaries of the simulation environment.

    Attributes:
        max_x: Maximum x-coordinate.
        max_y: Maximum y-coordinate.
        max_z: Maximum z-coordinate (altitude).
        min_x: Minimum x-coordinate (default: 0).
        min_y: Minimum y-coordinate (default: 0).
        min_z: Minimum z-coordinate (default: 0).
    """

    max_x: float
    max_y: float
    max_z: float
    min_x: float = 0.0
    min_y: float = 0.0
    min_z: float = 0.0

    def contains(self, location: Location) -> bool:
        """Check if a location is within the boundary.

        Args:
            location: Location to check.

        Returns:
            True if location is within bounds, False otherwise.
        """
        return (
            self.min_x <= location.x <= self.max_x
            and self.min_y <= location.y <= self.max_y
            and self.min_z <= location.z <= self.max_z
        )

    def clamp(self, location: Location) -> Location:
        """Clamp a location to be within the boundary.

        Args:
            location: Location to clamp.

        Returns:
            New location clamped to boundary.
        """
        return Location(
            x=max(self.min_x, min(self.max_x, location.x)),
            y=max(self.min_y, min(self.max_y, location.y)),
            z=max(self.min_z, min(self.max_z, location.z)),
        )

    def random_location(self, altitude: float = 0.0) -> Location:
        """Generate a random location within the boundary.

        Args:
            altitude: Fixed altitude for the location.

        Returns:
            Random location within bounds.
        """
        return Location.random_within(
            max_x=self.max_x,
            max_y=self.max_y,
            altitude=altitude,
            min_x=self.min_x,
            min_y=self.min_y,
        )
