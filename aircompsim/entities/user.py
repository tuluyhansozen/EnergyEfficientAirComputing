"""User entity definitions.

This module provides User classes for modeling end-users
in the air computing simulation environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Set

import numpy as np

from aircompsim.entities.location import Location
from aircompsim.entities.task import Application

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Represents an end-user in the simulation.

    Users have a location, mobility state, and set of applications
    that generate computational tasks.

    Attributes:
        location: Current user location.
        user_id: Unique user identifier.
        is_moving: Whether user is currently in motion.
        applications: Set of user applications.
        trajectory: History of visited locations.
        city: Optional city/region identifier.

    Example:
        >>> user = User(location=Location(100, 200, 0))
        >>> user.add_application(my_app)
        >>> new_loc = user.get_next_location(radius=50)
    """

    location: Location
    user_id: int = field(default=0)
    is_moving: bool = False
    qoe: float = 0.0
    applications: Set[Application] = field(default_factory=set)
    trajectory: List[Location] = field(default_factory=list)
    city: str = ""
    speed: float = 2.0  # Default walking speed in m/s

    # Class-level registry
    _id_counter: ClassVar[int] = 0
    _all_users: ClassVar[List[User]] = []

    def __post_init__(self) -> None:
        """Initialize user with unique ID."""
        User._id_counter += 1
        object.__setattr__(self, "user_id", User._id_counter)
        User._all_users.append(self)
        self.trajectory.append(self.location.copy())

        logger.debug(f"User {self.user_id} created at {self.location}")

    def __eq__(self, other: object) -> bool:
        """Check equality by ID."""
        if not isinstance(other, User):
            return NotImplemented
        return self.user_id == other.user_id

    def __hash__(self) -> int:
        """Hash by ID."""
        return hash(("User", self.user_id))

    @property
    def id(self) -> int:
        """Get user ID."""
        return self.user_id

    @property
    def current_location(self) -> Location:
        """Get current location (alias for location)."""
        return self.location

    @current_location.setter
    def current_location(self, loc: Location) -> None:
        """Set current location."""
        self.location = loc

    def add_application(self, app: Application) -> None:
        """Add an application to this user.

        Args:
            app: Application to add.
        """
        self.applications.add(app)
        app.user_id = self.user_id
        logger.debug(f"User {self.user_id}: Added application {app.app_id}")

    def remove_application(self, app: Application) -> bool:
        """Remove an application from this user.

        Args:
            app: Application to remove.

        Returns:
            True if removed, False if not found.
        """
        if app in self.applications:
            self.applications.discard(app)
            logger.debug(f"User {self.user_id}: Removed application {app.app_id}")
            return True
        return False

    def get_location(self) -> Location:
        """Get current location."""
        return self.location

    def compute_movement_duration(self, destination: Location) -> float:
        """Calculate time to reach a destination.

        Args:
            destination: Target location.

        Returns:
            Movement duration in seconds.
        """
        distance = Location.euclidean_distance_2d(self.location, destination)
        return distance / self.speed

    def get_next_location(self, radius: float, boundary: Optional[tuple] = None) -> Location:
        """Generate a random next location within radius.

        Args:
            radius: Maximum distance to move.
            boundary: Optional (max_x, max_y) bounds.

        Returns:
            New random location.
        """
        new_x = np.random.uniform(self.location.x - radius, self.location.x + radius)
        new_y = np.random.uniform(self.location.y - radius, self.location.y + radius)

        if boundary:
            max_x, max_y = boundary
            new_x = max(0, min(max_x, new_x))
            new_y = max(0, min(max_y, new_y))

        return Location(x=new_x, y=new_y, z=0)

    def move_to(self, destination: Location) -> None:
        """Move user to a new location.

        Args:
            destination: New location.
        """
        self.location = destination
        self.trajectory.append(destination.copy())
        logger.debug(f"User {self.user_id} moved to {destination}")

    def start_moving(self) -> None:
        """Mark user as in motion."""
        self.is_moving = True

    def stop_moving(self) -> None:
        """Mark user as stationary."""
        self.is_moving = False

    def get_qoe(self) -> float:
        """Get user Quality of Experience.

        Computed as average QoS across all applications.

        Returns:
            Average QoE (0-100).
        """
        if not self.applications:
            return 0.0

        total_qos = sum(app.compute_qos() for app in self.applications)
        self.qoe = total_qos / len(self.applications)
        return self.qoe

    def get_statistics(self) -> dict:
        """Get user statistics summary."""
        return {
            "user_id": self.user_id,
            "location": str(self.location),
            "is_moving": self.is_moving,
            "num_applications": len(self.applications),
            "trajectory_length": len(self.trajectory),
            "qoe": self.get_qoe(),
        }

    @classmethod
    def reset_all(cls) -> None:
        """Reset all users."""
        cls._id_counter = 0
        cls._all_users.clear()
        logger.debug("User registry reset")

    @classmethod
    def get_all(cls) -> List[User]:
        """Get all users."""
        return cls._all_users.copy()

    @classmethod
    def remove(cls, user_id: int) -> bool:
        """Remove a user by ID.

        Args:
            user_id: User ID to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, user in enumerate(cls._all_users):
            if user.user_id == user_id:
                # Remove applications
                for app in list(user.applications):
                    Application.remove_application(app.app_id)
                cls._all_users.pop(i)
                logger.info(f"User {user_id} removed")
                return True
        return False

    @classmethod
    def get_user(cls, user_id: int) -> Optional[User]:
        """Get user by ID."""
        for user in cls._all_users:
            if user.user_id == user_id:
                return user
        return None


@dataclass
class MobileUser(User):
    """User with mobility capabilities.

    Extends User with speed and trajectory tracking
    specific to mobile users.
    """

    max_speed: float = 5.0  # Maximum speed in m/s

    def update_speed(self, new_speed: float) -> None:
        """Update user speed.

        Args:
            new_speed: New speed in m/s.
        """
        self.speed = min(new_speed, self.max_speed)


@dataclass
class FlyingUser(MobileUser):
    """User in the air (e.g., on aircraft or drone).

    Extends MobileUser with altitude tracking.
    """

    altitude: float = 0.0

    @property
    def current_location(self) -> Location:
        """Get current 3D location including altitude."""
        return Location(self.location.x, self.location.y, self.altitude)

    @current_location.setter
    def current_location(self, loc: Location) -> None:
        """Set 3D location."""
        self.location.x = loc.x
        self.location.y = loc.y
        self.altitude = loc.z


# Alias for backward compatibility
def get_all_users() -> List[User]:
    """Get all users."""
    return User.get_all()
