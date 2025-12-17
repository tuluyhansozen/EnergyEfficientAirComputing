"""Battery management and modeling.

This module provides battery models for UAVs including capacity,
charging, degradation, and state-of-charge management.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from aircompsim.energy.models import EnergyMode

logger = logging.getLogger(__name__)


@dataclass
class BatterySpecifications:
    """Battery specifications for modeling.
    
    Attributes:
        capacity: Nominal capacity in Wh.
        voltage: Nominal voltage in V.
        max_discharge_rate: Maximum C-rate for discharge.
        max_charge_rate: Maximum C-rate for charging.
        min_voltage: Minimum safe voltage in V.
        max_cycles: Expected cycle life.
    """
    
    capacity: float = 5935.0  # Wh (DJI TB60)
    voltage: float = 52.8
    max_discharge_rate: float = 2.0  # 2C
    max_charge_rate: float = 1.0     # 1C
    min_voltage: float = 40.0
    max_cycles: int = 400


@dataclass
class BatteryModel:
    """Lithium-polymer battery model with degradation.
    
    Models battery state-of-charge, voltage characteristics,
    and capacity degradation over charge cycles.
    
    Attributes:
        specs: Battery specifications.
        state_of_charge: Current charge level (0-100%).
        cycle_count: Number of charge cycles.
        health: Battery health factor (0-1).
        
    Example:
        >>> battery = BatteryModel()
        >>> battery.discharge(100)  # Remove 100J
        >>> print(f"SoC: {battery.state_of_charge:.1f}%")
    """
    
    specs: BatterySpecifications = field(default_factory=BatterySpecifications)
    state_of_charge: float = 100.0
    cycle_count: int = 0
    health: float = 1.0
    _temperature: float = 25.0  # Celsius
    
    def __post_init__(self) -> None:
        """Validate initial state."""
        self.state_of_charge = max(0.0, min(100.0, self.state_of_charge))
        self.health = max(0.0, min(1.0, self.health))
    
    @property
    def effective_capacity(self) -> float:
        """Get effective capacity considering health and temperature.
        
        Returns:
            Effective capacity in Wh.
        """
        # Temperature effect (capacity reduces at extreme temperatures)
        temp_factor = self._temperature_factor()
        
        # Health degradation
        return self.specs.capacity * self.health * temp_factor
    
    @property
    def available_energy(self) -> float:
        """Get available energy in Joules.
        
        Returns:
            Available energy considering SoC and health.
        """
        # Convert Wh to Joules (1 Wh = 3600 J)
        return self.effective_capacity * (self.state_of_charge / 100.0) * 3600
    
    @property
    def energy_mode(self) -> EnergyMode:
        """Get current energy mode based on state of charge."""
        return EnergyMode.HIGH if self.state_of_charge > 70 else \
               EnergyMode.MEDIUM if self.state_of_charge > 30 else \
               EnergyMode.LOW if self.state_of_charge > 10 else \
               EnergyMode.CRITICAL
    
    def _temperature_factor(self) -> float:
        """Calculate temperature effect on capacity.
        
        Returns:
            Capacity factor (0-1) based on temperature.
        """
        if 15 <= self._temperature <= 35:
            return 1.0
        elif self._temperature < 15:
            # Cold temperature reduces capacity
            return max(0.5, 1.0 - 0.02 * (15 - self._temperature))
        else:
            # Hot temperature slightly reduces capacity
            return max(0.8, 1.0 - 0.01 * (self._temperature - 35))
    
    def discharge(self, energy_joules: float) -> bool:
        """Discharge the battery.
        
        Args:
            energy_joules: Energy to discharge in Joules.
            
        Returns:
            True if discharge successful, False if insufficient energy.
        """
        if energy_joules < 0:
            logger.warning(f"Negative discharge requested: {energy_joules}")
            return False
        
        if energy_joules > self.available_energy:
            logger.warning(
                f"Insufficient battery: requested {energy_joules:.1f}J, "
                f"available {self.available_energy:.1f}J"
            )
            # Drain to zero instead of failing
            self.state_of_charge = 0.0
            return False
        
        # Convert Joules to percentage
        full_capacity_joules = self.effective_capacity * 3600
        percentage_used = (energy_joules / full_capacity_joules) * 100
        self.state_of_charge = max(0.0, self.state_of_charge - percentage_used)
        
        logger.debug(
            f"Battery discharged {energy_joules:.1f}J, SoC: {self.state_of_charge:.1f}%"
        )
        return True
    
    def charge(self, duration_seconds: float, charge_rate: float = 1.0) -> float:
        """Charge the battery.
        
        Args:
            duration_seconds: Charging duration in seconds.
            charge_rate: Charging rate as fraction of max (0-1).
            
        Returns:
            Energy added in Joules.
        """
        if self.state_of_charge >= 100.0:
            return 0.0
        
        # Calculate energy added
        power = self.specs.capacity * self.specs.max_charge_rate * charge_rate
        energy_wh = power * (duration_seconds / 3600)
        energy_joules = energy_wh * 3600
        
        # Convert to SoC percentage
        full_capacity_joules = self.effective_capacity * 3600
        percentage_added = (energy_joules / full_capacity_joules) * 100
        
        old_soc = self.state_of_charge
        self.state_of_charge = min(100.0, self.state_of_charge + percentage_added)
        actual_added = self.state_of_charge - old_soc
        
        # Track charging for cycle counting
        if old_soc < 20 and self.state_of_charge > 80:
            self.cycle_count += 1
            self._update_health()
        
        actual_energy = (actual_added / 100) * full_capacity_joules
        logger.debug(
            f"Battery charged {actual_energy:.1f}J, SoC: {self.state_of_charge:.1f}%"
        )
        return actual_energy
    
    def charge_to_full(self) -> tuple[float, float]:
        """Charge battery to 100%.
        
        Returns:
            Tuple of (time_required_seconds, energy_added_joules).
        """
        if self.state_of_charge >= 100.0:
            return (0.0, 0.0)
        
        # Energy needed
        full_capacity_joules = self.effective_capacity * 3600
        current_energy = (self.state_of_charge / 100) * full_capacity_joules
        energy_needed = full_capacity_joules - current_energy
        
        # Time to charge (in hours, then convert to seconds)
        power = self.specs.capacity * self.specs.max_charge_rate  # W
        time_hours = (energy_needed / 3600) / power  # Wh / W = hours
        time_seconds = time_hours * 3600
        
        # Perform charge
        old_soc = self.state_of_charge
        self.state_of_charge = 100.0
        self.cycle_count += 1
        self._update_health()
        
        logger.info(f"Battery fully charged: {old_soc:.1f}% -> 100%")
        return (time_seconds, energy_needed)
    
    def _update_health(self) -> None:
        """Update battery health based on cycle count."""
        # Linear degradation model
        # After max_cycles, health drops to ~80%
        degradation = 0.2 * (self.cycle_count / self.specs.max_cycles)
        self.health = max(0.5, 1.0 - degradation)
        
        logger.debug(f"Battery health updated: {self.health:.2f} ({self.cycle_count} cycles)")
    
    def can_complete_mission(
        self,
        estimated_energy: float,
        reserve_margin: float = 0.2
    ) -> bool:
        """Check if battery can complete a mission.
        
        Args:
            estimated_energy: Estimated mission energy in Joules.
            reserve_margin: Reserve margin as fraction (default 20%).
            
        Returns:
            True if mission can be completed with reserve.
        """
        required_energy = estimated_energy * (1 + reserve_margin)
        return self.available_energy >= required_energy
    
    def estimate_remaining_flight_time(
        self,
        power_consumption: float
    ) -> float:
        """Estimate remaining flight time.
        
        Args:
            power_consumption: Average power consumption in Watts.
            
        Returns:
            Estimated remaining time in seconds.
        """
        if power_consumption <= 0:
            return float('inf')
        
        # Available energy in Joules / Power in Watts = Time in seconds
        return self.available_energy / power_consumption
    
    def get_status(self) -> dict:
        """Get battery status summary."""
        return {
            "state_of_charge": self.state_of_charge,
            "health": self.health,
            "cycle_count": self.cycle_count,
            "effective_capacity": self.effective_capacity,
            "available_energy_j": self.available_energy,
            "energy_mode": self.energy_mode.name,
            "temperature": self._temperature
        }
    
    def set_temperature(self, temperature: float) -> None:
        """Set battery temperature.
        
        Args:
            temperature: Temperature in Celsius.
        """
        self._temperature = max(-20, min(60, temperature))
        logger.debug(f"Battery temperature set to {self._temperature}°C")
    
    def reset(self, full_charge: bool = True) -> None:
        """Reset battery state.
        
        Args:
            full_charge: If True, reset to 100% charge.
        """
        if full_charge:
            self.state_of_charge = 100.0
        self.cycle_count = 0
        self.health = 1.0
        self._temperature = 25.0
        logger.debug("Battery reset")
