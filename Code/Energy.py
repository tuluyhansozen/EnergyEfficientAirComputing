class EnergyModel:
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, delta=0.1): #TODO: coefficient calculations are NOT finalized yet
        """
        Energy model coefficients:
        alpha: flight energy per unit distance * velocity^2
        beta: hover energy per second
        gamma: computation energy per second
        delta: communication energy per second
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_flight_energy(self, distance, velocity):
        return self.alpha * distance * (velocity ** 2)
    
    def compute_hover_energy(self, hover_time):
        return self.beta * hover_time
    
    def compute_computation_energy(self, computation_time):
        return self.gamma * computation_time
    
    def compute_communication_energy(self, communication_time):
        return self.delta * communication_time
    
    def compute_total_energy(self, distance, velocity, hover_time, computation_time, communication_time):
        flight_energy = self.compute_flight_energy(distance, velocity)
        hover_energy = self.compute_hover_energy(hover_time)
        computation_energy = self.compute_computation_energy(computation_time)
        communication_energy = self.compute_communication_energy(communication_time)
        
        return flight_energy + hover_energy + computation_energy + communication_energy
    
    def summary(self, distance, velocity, hover_time, computation_time, communication_time):
        total_energy = self.compute_total_energy(distance, velocity, hover_time, computation_time, communication_time)
        return {
            "Total Energy": total_energy,
            "Flight Energy": self.compute_flight_energy(distance, velocity),
            "Hover Energy": self.compute_hover_energy(hover_time),
            "Computation Energy": self.compute_computation_energy(computation_time),
            "Communication Energy": self.compute_communication_energy(communication_time)
        }
    
class ChargingStation:
    def __init__(self, location, capacity=2, charging_rate=1):
        """
        Represents a fixed-position UAV charging station.
        :param location: tuple (x, y)
        :param capacity: maximum number of UAVs it can charge at once
        :param charging_rate: energy percentage added per time unit
        """
        self.location = location
        self.capacity = capacity
        self.occupied_slots = 0
        self.charging_rate = charging_rate

    def is_available(self):
        return self.occupied_slots < self.capacity
    
    def start_charging(self):
        if self.is_available():
            self.occupied_slots += 1
            return True
        return False
    
    def stop_charging(self):
        if self.occupied_slots > 0:
            self.occupied_slots -= 1
            return True
        return False
    
    def charge(self, current_energy):
        """
        Simulate one unit of charging. Returns updated energy level.
        """
        return min(100, current_energy + self.charging_rate)

    def get_status(self):
        return {
            "Location": self.location,
            "Occupied Slots": self.occupied_slots,
            "Capacity": self.capacity
        }
    
# UAV HELPERS

def should_accept_task(uav, estimated_energy_cost):
    """
    Returns True if UAV can accept the task based on its energy level and mode.
    """
    if uav.energy_mode == "Critical":
        return False
    elif uav.energy_mode == "Low" and estimated_energy_cost > 10:
        return False
    return uav.batteryLevel >= estimated_energy_cost + 5  # keeps some buffer

def should_return_to_charge(uav):
    """
    Determines if UAV should go to nearest charging station.
    """
    return uav.energy_mode == "Critical"

def energy_based_behavior_summary(uav):
    return f"[UAV {uav.id}] Energy: {uav.batteryLevel:.1f}%, Mode: {uav.energy_mode}"
