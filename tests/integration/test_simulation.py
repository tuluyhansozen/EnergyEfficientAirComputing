"""Integration tests for the simulation system."""

import pytest
import numpy as np

from aircompsim.entities.location import Location, SimulationBoundary
from aircompsim.entities.server import EdgeServer, UAV, CloudServer
from aircompsim.entities.user import User
from aircompsim.entities.task import (
    ApplicationType, Application, Task, OffloadEntity
)
from aircompsim.energy.models import EnergyModel, EnergyMode
from aircompsim.config.settings import SimulationConfig


class TestServerUserIntegration:
    """Integration tests for server and user interactions."""
    
    @pytest.fixture(autouse=True)
    def reset_registries(self):
        """Reset all registries before each test."""
        EdgeServer.reset_all()
        UAV.reset_all()
        User.reset_all()
        Application.reset_all()
        ApplicationType.clear_registry()
        yield
        # Cleanup after test
        EdgeServer.reset_all()
        UAV.reset_all()
        User.reset_all()
        Application.reset_all()
        ApplicationType.clear_registry()
    
    def test_user_in_edge_coverage(self):
        """Test user location within edge server coverage."""
        edge = EdgeServer(
            capacity=1000,
            location=Location(100, 100, 0),
            radius=50,
            power_consumption=100
        )
        
        # User inside coverage
        user_inside = User(location=Location(110, 110, 0))
        assert edge.is_in_coverage(user_inside.location)
        
        # User outside coverage
        user_outside = User(location=Location(200, 200, 0))
        assert not edge.is_in_coverage(user_outside.location)
    
    def test_user_covered_by_uav(self):
        """Test user coverage by flying UAV."""
        uav = UAV(
            capacity=500,
            location=Location(150, 150, 200),
            radius=100,
            power_consumption=50
        )
        
        # User below UAV
        user = User(location=Location(160, 160, 0))
        assert uav.is_in_coverage(user.location)
    
    def test_multiple_servers_coverage(self):
        """Test user covered by multiple servers."""
        edge1 = EdgeServer(
            capacity=1000,
            location=Location(100, 100, 0),
            radius=80,
            power_consumption=100
        )
        edge2 = EdgeServer(
            capacity=1000,
            location=Location(150, 100, 0),
            radius=80,
            power_consumption=100
        )
        
        # User in overlap zone
        user = User(location=Location(125, 100, 0))
        
        covering_servers = [
            s for s in EdgeServer.get_all()
            if s.is_in_coverage(user.location)
        ]
        
        assert len(covering_servers) == 2


class TestTaskLifecycle:
    """Integration tests for task creation and processing."""
    
    @pytest.fixture(autouse=True)
    def reset_registries(self):
        """Reset all registries before each test."""
        EdgeServer.reset_all()
        User.reset_all()
        Application.reset_all()
        ApplicationType.clear_registry()
        Task.reset_all()
        yield
        EdgeServer.reset_all()
        User.reset_all()
        Application.reset_all()
        ApplicationType.clear_registry()
        Task.reset_all()
    
    def test_task_generation_and_completion(self):
        """Test full task lifecycle from creation to completion."""
        # Create app type
        app_type = ApplicationType(
            name="TestApp",
            cpu_cycle=100,
            worst_delay=1.0,
            best_delay=0.1,
            interarrival_time=10
        )
        
        # Create user with application
        user = User(location=Location(100, 100, 0))
        app = Application(app_type=app_type, start_time=0)
        user.add_application(app)
        
        # Create server
        edge = EdgeServer(
            capacity=1000,
            location=Location(100, 100, 0),
            radius=100,
            power_consumption=100
        )
        
        # Generate task
        task = app.generate_task(user)
        
        assert task.app == app
        assert task.user == user
        assert task.offload_entity == OffloadEntity.USER_TO_EDGE
        
        # Process task
        processing_delay = edge.get_processing_delay(task)
        assert processing_delay > 0
        
        # Complete task
        end_time = task.creation_time + processing_delay
        task.complete(end_time, edge)
        
        assert task.end_time == end_time
        assert task.processed_server == edge
        assert task.qos >= 0  # QoS computed
    
    def test_task_qos_scoring(self):
        """Test QoS scoring based on latency."""
        app_type = ApplicationType(
            name="QoSTestApp",
            cpu_cycle=100,
            worst_delay=0.5,
            best_delay=0.1,
            interarrival_time=10
        )
        
        user = User(location=Location(0, 0, 0))
        app = Application(app_type=app_type, start_time=0)
        
        # Fast task (excellent)
        task_fast = Task(app=app, user=user, creation_time=0)
        task_fast.end_time = 0.05  # Under best_delay
        task_fast.compute_qos()
        assert task_fast.qos == 100
        
        # Medium task (acceptable)
        task_medium = Task(app=app, user=user, creation_time=1)
        task_medium.end_time = 1.3  # Between best and worst
        task_medium.compute_qos()
        assert task_medium.qos == 50
        
        # Slow task (failed)
        task_slow = Task(app=app, user=user, creation_time=2)
        task_slow.end_time = 3.0  # Over worst_delay
        task_slow.compute_qos()
        assert task_slow.qos == 0


class TestUAVEnergyIntegration:
    """Integration tests for UAV energy consumption."""
    
    @pytest.fixture(autouse=True)
    def reset_registries(self):
        """Reset all registries before each test."""
        UAV.reset_all()
        yield
        UAV.reset_all()
    
    def test_uav_flight_energy_consumption(self):
        """Test UAV battery drain during flight."""
        uav = UAV(
            capacity=500,
            location=Location(0, 0, 200),
            radius=100,
            power_consumption=50,
            battery_level=100
        )
        
        initial_battery = uav.battery_level
        
        # Fly 100 meters
        uav.consume_flight_energy(distance=100, velocity=10)
        
        assert uav.battery_level < initial_battery
        assert uav.energy_tracker.flight_energy > 0
    
    def test_uav_energy_mode_transitions(self):
        """Test UAV energy mode changes with battery drain."""
        uav = UAV(
            capacity=500,
            location=Location(0, 0, 200),
            radius=100,
            power_consumption=50,
            battery_level=100
        )
        
        assert uav.energy_mode == EnergyMode.HIGH
        
        # Drain to medium
        uav.battery_level = 50
        assert uav.energy_mode == EnergyMode.MEDIUM
        
        # Drain to low
        uav.battery_level = 20
        assert uav.energy_mode == EnergyMode.LOW
        
        # Drain to critical
        uav.battery_level = 5
        assert uav.energy_mode == EnergyMode.CRITICAL
        assert not uav.can_accept_task()
    
    def test_uav_recharge(self):
        """Test UAV recharging."""
        uav = UAV(
            capacity=500,
            location=Location(0, 0, 200),
            radius=100,
            power_consumption=50,
            battery_level=20
        )
        
        uav.recharge(50)
        assert uav.battery_level == 70
        
        uav.recharge(100)  # Exceeds max
        assert uav.battery_level == 100


class TestSimulationConfig:
    """Integration tests for configuration system."""
    
    def test_config_creation(self):
        """Test simulation config creation."""
        config = SimulationConfig(
            time_limit=2000,
            user_count=50
        )
        
        assert config.time_limit == 2000
        assert config.user_count == 50
        assert config.uav.count == 5  # Default
    
    def test_config_nested_modification(self):
        """Test modifying nested config."""
        config = SimulationConfig()
        
        config.uav.count = 10
        config.uav.capacity = 1000
        config.energy.use_physics_model = True
        
        assert config.uav.count == 10
        assert config.energy.use_physics_model is True
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = SimulationConfig(
            time_limit=1000,
            user_count=20
        )
        
        data = config.to_dict()
        
        assert data['time_limit'] == 1000
        assert data['user_count'] == 20
        assert 'uav' in data
        assert 'energy' in data


class TestEntityRegistries:
    """Integration tests for entity registration."""
    
    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset all registries."""
        EdgeServer.reset_all()
        UAV.reset_all()
        User.reset_all()
        Application.reset_all()
        Task.reset_all()
        yield
        EdgeServer.reset_all()
        UAV.reset_all()
        User.reset_all()
        Application.reset_all()
        Task.reset_all()
    
    def test_edge_server_registry(self):
        """Test edge server auto-registration."""
        edge1 = EdgeServer(
            capacity=1000,
            location=Location(0, 0, 0),
            radius=100,
            power_consumption=100
        )
        edge2 = EdgeServer(
            capacity=1000,
            location=Location(100, 100, 0),
            radius=100,
            power_consumption=100
        )
        
        all_edges = EdgeServer.get_all()
        assert len(all_edges) == 2
        assert edge1 in all_edges
        assert edge2 in all_edges
    
    def test_uav_registry(self):
        """Test UAV auto-registration."""
        for _ in range(5):
            UAV(
                capacity=500,
                location=Location(np.random.uniform(0, 400), np.random.uniform(0, 400), 200),
                radius=100,
                power_consumption=50
            )
        
        assert len(UAV.get_all()) == 5
    
    def test_registry_reset(self):
        """Test registry reset clears entities."""
        EdgeServer(
            capacity=1000,
            location=Location(0, 0, 0),
            radius=100,
            power_consumption=100
        )
        
        assert len(EdgeServer.get_all()) == 1
        
        EdgeServer.reset_all()
        
        assert len(EdgeServer.get_all()) == 0
