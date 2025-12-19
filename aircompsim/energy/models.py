"""Energy models for UAVs, edge servers, and cloud computing.

This module provides physics-based energy consumption models for
air computing environments, including flight, hover, computation,
and communication energy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EnergyMode(Enum):
    """Energy mode based on battery level.

    Determines UAV behavior regarding task acceptance and movement.
    """

    HIGH = auto()  # > 70% - Normal operation
    MEDIUM = auto()  # 30-70% - Conservative operation
    LOW = auto()  # 10-30% - Return to charge recommended
    CRITICAL = auto()  # < 10% - Emergency, must charge


@dataclass
class EnergyCoefficients:
    """Energy model coefficients for power calculations.

    These coefficients are based on empirical models for medium-sized
    UAVs (e.g., DJI Matrice series).

    Attributes:
        alpha: Flight energy coefficient (J/(m·(m/s)²)).
        beta: Hover power (W).
        gamma: Computation power (W).
        delta: Communication power (W).

    References:
        - "Energy-Efficient UAV Communication", IEEE Trans. Wireless, 2019
        - "Optimal UAV Deployment", IEEE Access, 2020
    """

    alpha: float = 0.05  # Flight: α × distance × velocity²
    beta: float = 3.0  # Hover: β × time
    gamma: float = 20.0  # Computation: γ × time
    delta: float = 10.0  # Communication: δ × time


@dataclass
class UAVSpecifications:
    """Physical specifications for UAV energy modeling.

    Based on typical specifications for medium-sized commercial UAVs.

    Attributes:
        mass: UAV mass in kg.
        blade_radius: Propeller blade radius in meters.
        blade_count: Number of propeller blades.
        air_density: Air density in kg/m³.
        propeller_efficiency: Propeller efficiency (0-1).
        max_speed: Maximum horizontal speed in m/s.
        max_payload: Maximum payload capacity in kg.
        battery_capacity: Battery capacity in Wh.
        battery_voltage: Battery voltage in V.
    """

    mass: float = 6.3
    blade_radius: float = 0.2
    blade_count: int = 4
    air_density: float = 1.225
    propeller_efficiency: float = 0.8
    max_speed: float = 23.0
    max_payload: float = 2.7
    battery_capacity: float = 5935.0  # Wh (DJI TB60 battery)
    battery_voltage: float = 52.8


class EnergyModel:
    """Energy consumption model for simulation entities.

    Provides methods to calculate energy consumption for various
    activities including flight, hovering, computation, and communication.

    Attributes:
        coefficients: Energy calculation coefficients.
        specs: UAV physical specifications (optional).
        use_physics_model: If True, use physics-based calculations.

    Example:
        >>> model = EnergyModel()
        >>> flight_energy = model.compute_flight_energy(distance=100, velocity=10)
        >>> print(f"Flight energy: {flight_energy:.2f} J")
        Flight energy: 500.00 J
    """

    def __init__(
        self,
        coefficients: EnergyCoefficients | None = None,
        specs: UAVSpecifications | None = None,
        use_physics_model: bool = False,
    ) -> None:
        """Initialize energy model.

        Args:
            coefficients: Energy coefficients for simplified model.
            specs: UAV specifications for physics-based model.
            use_physics_model: Whether to use physics-based calculations.
        """
        self.coefficients = coefficients or EnergyCoefficients()
        self.specs = specs or UAVSpecifications()
        self.use_physics_model = use_physics_model

        logger.debug(
            f"EnergyModel initialized: physics={use_physics_model}, " f"coeffs={self.coefficients}"
        )

    def compute_flight_energy(
        self,
        distance: float,
        velocity: float,
        payload: float = 0.0,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0,
    ) -> float:
        """Calculate energy consumed during UAV flight.

        Uses either simplified model (E = α × d × v²) or physics-based
        model depending on configuration.

        Args:
            distance: Flight distance in meters.
            velocity: Flight velocity in m/s.
            payload: Additional payload mass in kg.
            wind_speed: Wind speed in m/s (for physics model).
            wind_direction: Wind direction relative to flight (radians).

        Returns:
            Energy consumed in Joules.

        Raises:
            ValueError: If distance or velocity is negative.
        """
        if distance < 0:
            raise ValueError(f"Distance cannot be negative: {distance}")
        if velocity < 0:
            raise ValueError(f"Velocity cannot be negative: {velocity}")

        if distance == 0:
            return 0.0

        if self.use_physics_model:
            return self._physics_flight_energy(
                distance, velocity, payload, wind_speed, wind_direction
            )

        # Simplified model: E = α × d × v²
        energy = self.coefficients.alpha * distance * (velocity**2)
        logger.debug(f"Flight energy: {energy:.2f}J for {distance:.1f}m at {velocity:.1f}m/s")
        return energy

    def _physics_flight_energy(
        self,
        distance: float,
        velocity: float,
        payload: float,
        wind_speed: float,
        wind_direction: float,
    ) -> float:
        """Physics-based flight energy calculation.

        Based on blade element momentum theory and empirical corrections.
        """
        # Total weight
        total_mass = self.specs.mass + payload
        weight = total_mass * 9.81  # N

        # Effective velocity considering wind
        effective_velocity = velocity - wind_speed * math.cos(wind_direction)
        effective_velocity = max(0.1, effective_velocity)  # Prevent division by zero

        # Flight time
        flight_time = distance / effective_velocity

        # Induced power (simplified momentum theory)
        disk_area = math.pi * (self.specs.blade_radius**2) * self.specs.blade_count
        induced_velocity = math.sqrt(weight / (2 * self.specs.air_density * disk_area))
        induced_power = weight * induced_velocity / self.specs.propeller_efficiency

        # Profile power (blade drag)
        profile_power = 0.012 * self.specs.air_density * disk_area * (velocity**3)

        # Parasite power (fuselage drag)
        drag_area = 0.1  # m² (typical for medium UAV)
        parasite_power = 0.5 * self.specs.air_density * drag_area * (velocity**3)

        # Total power
        total_power = induced_power + profile_power + parasite_power

        # Energy = Power × Time
        energy = total_power * flight_time

        logger.debug(
            f"Physics flight energy: {energy:.2f}J "
            f"(P_ind={induced_power:.1f}W, P_prof={profile_power:.1f}W, "
            f"P_para={parasite_power:.1f}W, t={flight_time:.1f}s)"
        )

        return energy

    def compute_hover_energy(self, duration: float) -> float:
        """Calculate energy consumed while hovering.

        Args:
            duration: Hover duration in seconds.

        Returns:
            Energy consumed in Joules.

        Raises:
            ValueError: If duration is negative.
        """
        if duration < 0:
            raise ValueError(f"Duration cannot be negative: {duration}")

        energy = self.coefficients.beta * duration
        logger.debug(f"Hover energy: {energy:.2f}J for {duration:.1f}s")
        return energy

    def compute_computation_energy(self, duration: float) -> float:
        """Calculate energy consumed during computation.

        Args:
            duration: Computation duration in seconds.

        Returns:
            Energy consumed in Joules.

        Raises:
            ValueError: If duration is negative.
        """
        if duration < 0:
            raise ValueError(f"Duration cannot be negative: {duration}")

        energy = self.coefficients.gamma * duration
        logger.debug(f"Computation energy: {energy:.2f}J for {duration:.1f}s")
        return energy

    def compute_communication_energy(self, duration: float) -> float:
        """Calculate energy consumed during communication.

        Args:
            duration: Communication duration in seconds.

        Returns:
            Energy consumed in Joules.

        Raises:
            ValueError: If duration is negative.
        """
        if duration < 0:
            raise ValueError(f"Duration cannot be negative: {duration}")

        energy = self.coefficients.delta * duration
        logger.debug(f"Communication energy: {energy:.2f}J for {duration:.1f}s")
        return energy

    def compute_total_energy(
        self,
        distance: float,
        velocity: float,
        hover_time: float,
        computation_time: float,
        communication_time: float,
    ) -> float:
        """Calculate total energy consumption for a mission.

        Args:
            distance: Flight distance in meters.
            velocity: Flight velocity in m/s.
            hover_time: Hover duration in seconds.
            computation_time: Computation duration in seconds.
            communication_time: Communication duration in seconds.

        Returns:
            Total energy consumed in Joules.
        """
        flight = self.compute_flight_energy(distance, velocity)
        hover = self.compute_hover_energy(hover_time)
        computation = self.compute_computation_energy(computation_time)
        communication = self.compute_communication_energy(communication_time)

        total = flight + hover + computation + communication

        logger.info(
            f"Total energy: {total:.2f}J "
            f"(flight={flight:.1f}, hover={hover:.1f}, "
            f"comp={computation:.1f}, comm={communication:.1f})"
        )

        return total

    def get_energy_summary(
        self,
        distance: float,
        velocity: float,
        hover_time: float,
        computation_time: float,
        communication_time: float,
    ) -> dict[str, float]:
        """Get detailed energy breakdown.

        Args:
            distance: Flight distance in meters.
            velocity: Flight velocity in m/s.
            hover_time: Hover duration in seconds.
            computation_time: Computation duration in seconds.
            communication_time: Communication duration in seconds.

        Returns:
            Dictionary with energy breakdown by category.
        """
        flight = self.compute_flight_energy(distance, velocity)
        hover = self.compute_hover_energy(hover_time)
        computation = self.compute_computation_energy(computation_time)
        communication = self.compute_communication_energy(communication_time)

        return {
            "flight_energy": flight,
            "hover_energy": hover,
            "computation_energy": computation,
            "communication_energy": communication,
            "total_energy": flight + hover + computation + communication,
        }

    @staticmethod
    def determine_energy_mode(battery_level: float) -> EnergyMode:
        """Determine energy mode based on battery level.

        Args:
            battery_level: Battery level as percentage (0-100).

        Returns:
            Appropriate EnergyMode based on battery level.
        """
        if battery_level > 70:
            return EnergyMode.HIGH
        elif battery_level > 30:
            return EnergyMode.MEDIUM
        elif battery_level > 10:
            return EnergyMode.LOW
        else:
            return EnergyMode.CRITICAL


@dataclass
class EnergyTracker:
    """Tracks cumulative energy consumption for an entity.

    Provides methods to record and query energy usage over time.

    Attributes:
        total_consumed: Total energy consumed in Joules.
        flight_energy: Energy consumed in flight.
        hover_energy: Energy consumed hovering.
        computation_energy: Energy consumed in computation.
        communication_energy: Energy consumed in communication.
    """

    total_consumed: float = 0.0
    flight_energy: float = 0.0
    hover_energy: float = 0.0
    computation_energy: float = 0.0
    communication_energy: float = 0.0
    _history: list[tuple[float, str, float]] = field(default_factory=list)

    def record(self, amount: float, energy_type: str, timestamp: float = 0.0) -> None:
        """Record energy consumption.

        Args:
            amount: Energy amount in Joules.
            energy_type: Type of energy (flight, hover, computation, communication).
            timestamp: Simulation time of consumption.
        """
        self.total_consumed += amount

        if energy_type == "flight":
            self.flight_energy += amount
        elif energy_type == "hover":
            self.hover_energy += amount
        elif energy_type == "computation":
            self.computation_energy += amount
        elif energy_type == "communication":
            self.communication_energy += amount

        self._history.append((timestamp, energy_type, amount))

    def get_breakdown(self) -> dict[str, float]:
        """Get energy consumption breakdown by type."""
        return {
            "total": self.total_consumed,
            "flight": self.flight_energy,
            "hover": self.hover_energy,
            "computation": self.computation_energy,
            "communication": self.communication_energy,
        }

    def reset(self) -> None:
        """Reset all energy tracking."""
        self.total_consumed = 0.0
        self.flight_energy = 0.0
        self.hover_energy = 0.0
        self.computation_energy = 0.0
        self.communication_energy = 0.0
        self._history.clear()
