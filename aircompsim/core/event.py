"""Event system for discrete event simulation.

This module provides the Event class and EventType enum for managing
simulation events in the event queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aircompsim.entities.location import Location
    from aircompsim.entities.task import Task
    from aircompsim.entities.user import User
    from aircompsim.entities.server import UAV


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the simulation.

    Attributes:
        OFFLOAD: Task offloading from user to server.
        PROCESS: Task processing completion.
        RETURNED: Task result returned to user.
        USER_MOVE: User movement step.
        USER_STOP: User reached destination.
        UAV_MOVE: UAV movement step.
        UAV_STOP: UAV reached destination.
        STATE: DRL state update event.
        CHARGING_START: UAV started charging.
        CHARGING_COMPLETE: UAV finished charging.
        SERVER_FAILURE: Server failure event.
        SERVER_RECOVERY: Server recovery event.
    """

    OFFLOAD = auto()
    PROCESS = auto()
    RETURNED = auto()
    USER_MOVE = auto()
    USER_STOP = auto()
    UAV_MOVE = auto()
    UAV_STOP = auto()
    STATE = auto()
    CHARGING_START = auto()
    CHARGING_COMPLETE = auto()
    SERVER_FAILURE = auto()
    SERVER_RECOVERY = auto()


@dataclass(order=True)
class Event:
    """Represents a discrete event in the simulation.

    Events are ordered by their scheduled time for use in a priority queue.
    The comparison is based solely on scheduled_time for heap operations.

    Attributes:
        scheduled_time: When the event should occur (simulation time).
        event_type: Type of the event.
        task: Associated task (if applicable).
        user: Associated user (if applicable).
        uav: Associated UAV (if applicable).
        location: Associated location (if applicable).
        data: Additional event data.

    Example:
        >>> event = Event(
        ...     scheduled_time=10.5,
        ...     event_type=EventType.OFFLOAD,
        ...     task=some_task
        ... )
        >>> import heapq
        >>> event_queue = []
        >>> heapq.heappush(event_queue, event)
    """

    scheduled_time: float
    event_type: EventType = field(compare=False)
    task: Optional[Task] = field(default=None, compare=False)
    user: Optional[User] = field(default=None, compare=False)
    uav: Optional[UAV] = field(default=None, compare=False)
    location: Optional[Location] = field(default=None, compare=False)
    data: Optional[dict[str, Any]] = field(default=None, compare=False)
    _id: int = field(default=0, compare=False, repr=False)

    # Class-level counter for unique IDs
    _id_counter: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Assign unique ID after initialization."""
        Event._id_counter += 1
        object.__setattr__(self, "_id", Event._id_counter)

    @property
    def id(self) -> int:
        """Get unique event ID."""
        return self._id

    def __str__(self) -> str:
        """Return human-readable event description."""
        return (
            f"Event(id={self._id}, type={self.event_type.name}, " f"time={self.scheduled_time:.2f})"
        )

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the event ID counter (for new simulation runs)."""
        cls._id_counter = 0
        logger.debug("Event counter reset")

    @classmethod
    def create_offload_event(cls, task: Task, time: float, location: Location) -> Event:
        """Factory method to create an offload event.

        Args:
            task: The task to be offloaded.
            time: Scheduled time for offloading.
            location: Location where offloading occurs.

        Returns:
            Configured Event instance.
        """
        return cls(scheduled_time=time, event_type=EventType.OFFLOAD, task=task, location=location)

    @classmethod
    def create_movement_event(
        cls,
        entity_type: str,
        entity: User | UAV,
        time: float,
        destination: Location,
        is_stop: bool = False,
    ) -> Event:
        """Factory method to create a movement event.

        Args:
            entity_type: Either 'user' or 'uav'.
            entity: The moving entity.
            time: Scheduled time for movement.
            destination: Target location.
            is_stop: If True, entity stops at destination.

        Returns:
            Configured Event instance.

        Raises:
            ValueError: If entity_type is not 'user' or 'uav'.
        """
        if entity_type == "user":
            event_type = EventType.USER_STOP if is_stop else EventType.USER_MOVE
            return cls(
                scheduled_time=time,
                event_type=event_type,
                user=entity,  # type: ignore
                location=destination,
            )
        elif entity_type == "uav":
            event_type = EventType.UAV_STOP if is_stop else EventType.UAV_MOVE
            return cls(
                scheduled_time=time,
                event_type=event_type,
                uav=entity,  # type: ignore
                location=destination,
            )
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    @classmethod
    def create_state_event(cls, time: float) -> Event:
        """Factory method to create a DRL state event.

        Args:
            time: Scheduled time for state observation.

        Returns:
            Configured Event instance.
        """
        return cls(scheduled_time=time, event_type=EventType.STATE)


class EventQueue:
    """Priority queue for simulation events.

    Wraps a heap-based priority queue with convenience methods
    for event management.

    Example:
        >>> queue = EventQueue()
        >>> queue.push(Event(scheduled_time=10, event_type=EventType.OFFLOAD))
        >>> queue.push(Event(scheduled_time=5, event_type=EventType.STATE))
        >>> event = queue.pop()  # Returns event at time 5
    """

    def __init__(self) -> None:
        """Initialize empty event queue."""
        self._queue: list[Event] = []
        self._event_count = 0

    def push(self, event: Event) -> None:
        """Add an event to the queue.

        Args:
            event: Event to add.
        """
        import heapq

        heapq.heappush(self._queue, event)
        self._event_count += 1
        logger.debug(f"Event pushed: {event}")

    def pop(self) -> Event:
        """Remove and return the next event.

        Returns:
            The event with the earliest scheduled time.

        Raises:
            IndexError: If the queue is empty.
        """
        import heapq

        if not self._queue:
            raise IndexError("pop from empty event queue")
        event = heapq.heappop(self._queue)
        logger.debug(f"Event popped: {event}")
        return event

    def peek(self) -> Optional[Event]:
        """View the next event without removing it.

        Returns:
            The next event, or None if queue is empty.
        """
        return self._queue[0] if self._queue else None

    def __len__(self) -> int:
        """Return number of events in queue."""
        return len(self._queue)

    def __bool__(self) -> bool:
        """Return True if queue has events."""
        return bool(self._queue)

    @property
    def total_events_processed(self) -> int:
        """Return total number of events ever added."""
        return self._event_count

    def clear(self) -> None:
        """Clear all events from the queue."""
        self._queue.clear()
        logger.debug("Event queue cleared")
