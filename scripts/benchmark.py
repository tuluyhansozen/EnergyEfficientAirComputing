"""Unified Benchmark Script for AirCompSim.

Runs all scenarios:
1. Paper Replication (Figs 4-6)
2. Advanced Scenarios (DRL, Charging, Mobility, Scheduling)
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aircompsim import Simulation, SimulationConfig
from aircompsim.energy.charging import ChargingStation, ChargingStationRegistry
from aircompsim.energy.scheduler import SchedulingStrategy
from aircompsim.entities.location import Location
from aircompsim.entities.server import UAV, EdgeServer
from aircompsim.entities.user import User


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    category: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    avg_latency: float
    avg_qos: float
    total_energy: float
    extra_metrics: Optional[dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


class UnifiedBenchmark:
    """Runs all benchmark scenarios."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(__file__).parent.parent / output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run_all(self):
        """Run all test suites."""
        print("=" * 70)
        print("AirCompSim Unified Benchmark Suite")
        print("=" * 70)
        print()

        # 1. Paper Replication (The core scenarios)
        self._run_paper_replication()

        # 2. DRL-based UAV positioning
        self._run_drl_positioning_tests()

        # 3. Charging station placement
        self._run_charging_station_tests()

        # 4. Mobility patterns
        self._run_mobility_tests()

        # 5. Scheduling algorithms
        self._run_scheduling_tests()

        return self.results

    def _run_simulation(
        self, name: str, category: str, config: SimulationConfig, setup_callback=None
    ) -> BenchmarkResult:
        """Run a single simulation and collect results."""
        print(f"  Running: {name}...")

        try:
            # Ensure deterministic workload
            config.workload_seed = 42

            sim = Simulation(config)
            sim.initialize()

            # Apply custom setup if provided
            if setup_callback:
                setup_callback(sim)

            results = sim.run()

            result = BenchmarkResult(
                name=name,
                category=category,
                total_tasks=results.total_tasks,
                successful_tasks=results.successful_tasks,
                success_rate=results.success_rate,
                avg_latency=results.avg_latency,
                avg_qos=results.avg_qos,
                total_energy=results.total_energy,
            )

            self.results.append(result)
            print(f"    ✓ Tasks: {results.total_tasks}, Success: {results.success_rate:.1%}")
            return result

        except Exception as e:
            print(f"    ✗ Error: {e}")
            result = BenchmarkResult(
                name=name,
                category=category,
                total_tasks=0,
                successful_tasks=0,
                success_rate=0,
                avg_latency=0,
                avg_qos=0,
                total_energy=0,
                extra_metrics={"error": str(e)},
            )
            self.results.append(result)
            return result

    # =========================================================================
    # 1. PAPER REPLICATION (Figs 4-6)
    # =========================================================================

    def _run_paper_replication(self):
        """Run scenarios to replicate paper figures."""
        print("\n" + "=" * 50)
        print("1. PAPER REPLICATION (Figs 4-6)")
        print("=" * 50)

        user_counts = [20, 40, 60, 80, 100]
        uav_counts = [0, 5, 10, 15, 20]

        for uavs in uav_counts:
            for users in user_counts:
                name = f"Users={users}, UAVs={uavs}"
                config = SimulationConfig(
                    time_limit=1000.0,
                    user_count=users,
                )
                config.uav.count = uavs
                config.edge.count = 4

                # We add extra metrics to help plotting later
                res = self._run_simulation(name, "Paper Replication", config)
                res.extra_metrics = {"users": users, "uavs": uavs}

    # =========================================================================
    # 2. DRL-BASED UAV POSITIONING TESTS
    # =========================================================================

    def _run_drl_positioning_tests(self):
        """Test different UAV positioning strategies."""
        print("\n" + "=" * 50)
        print("2. UAV POSITIONING STRATEGIES")
        print("=" * 50)

        strategies = [
            ("Random Positioning", self._random_uav_positions),
            ("Grid Positioning", self._grid_uav_positions),
            ("Edge-Centric", self._edge_centric_positions),
            ("User-Centric", self._user_centric_positions),
            ("Cluster-Based", self._cluster_positions),
        ]

        for name, position_func in strategies:
            config = SimulationConfig(time_limit=500, user_count=15)
            config.uav.count = 4
            config.edge.count = 4

            def setup(sim, func=position_func):
                func(sim)

            self._run_simulation(name, "UAV Positioning", config, setup)

    def _random_uav_positions(self, sim: Simulation):
        for uav in UAV.get_all():
            uav.location = Location(
                x=np.random.uniform(50, sim.boundary.max_x - 50),
                y=np.random.uniform(50, sim.boundary.max_y - 50),
                z=200,
            )

    def _grid_uav_positions(self, sim: Simulation):
        uavs = UAV.get_all()
        n = len(uavs)
        side = int(np.ceil(np.sqrt(n)))
        step_x = sim.boundary.max_x / (side + 1)
        step_y = sim.boundary.max_y / (side + 1)

        for i, uav in enumerate(uavs):
            row = i // side
            col = i % side
            uav.location = Location(x=step_x * (col + 1), y=step_y * (row + 1), z=200)

    def _edge_centric_positions(self, _sim: Simulation):
        uavs = UAV.get_all()
        edges = EdgeServer.get_all()
        for i, uav in enumerate(uavs):
            if i < len(edges):
                edge = edges[i]
                offset = 80
                angle = np.random.uniform(0, 2 * np.pi)
                uav.location = Location(
                    x=edge.location.x + offset * np.cos(angle),
                    y=edge.location.y + offset * np.sin(angle),
                    z=200,
                )

    def _user_centric_positions(self, _sim: Simulation):
        uavs = UAV.get_all()
        users = User.get_all()
        if not users:
            return
        for i, uav in enumerate(uavs):
            subset_size = max(1, len(users) // len(uavs))
            start = i * subset_size
            end = min(start + subset_size, len(users))
            if start < len(users):
                subset = users[start:end]
                avg_x = sum(u.location.x for u in subset) / len(subset)
                avg_y = sum(u.location.y for u in subset) / len(subset)
                uav.location = Location(x=avg_x, y=avg_y, z=200)

    def _cluster_positions(self, _sim: Simulation):
        uavs = UAV.get_all()
        clusters = [(100, 100), (300, 100), (100, 300), (300, 300), (200, 200)]
        for i, uav in enumerate(uavs):
            if i < len(clusters):
                x, y = clusters[i]
                uav.location = Location(x=x, y=y, z=200)

    # =========================================================================
    # 3. CHARGING STATION PLACEMENT TESTS
    # =========================================================================

    def _run_charging_station_tests(self):
        """Test charging station placement impact."""
        print("\n" + "=" * 50)
        print("3. CHARGING STATION PLACEMENT")
        print("=" * 50)

        placements = [
            ("No Charging Stations", 0, []),
            ("1 Station (Center)", 1, [(200, 200)]),
            ("2 Stations (Diagonal)", 2, [(100, 100), (300, 300)]),
            ("4 Stations (Corners)", 4, [(50, 50), (350, 50), (50, 350), (350, 350)]),
            ("4 Stations (Edges)", 4, [(200, 50), (350, 200), (200, 350), (50, 200)]),
        ]

        for name, _, positions in placements:
            config = SimulationConfig(time_limit=500, user_count=15)
            config.uav.count = 4
            config.uav.initial_battery = 60
            config.edge.count = 4

            def setup(_sim, pos=positions):
                registry = ChargingStationRegistry()
                registry.clear()
                for i, (x, y) in enumerate(pos):
                    station = ChargingStation(
                        station_id=i + 1,
                        location=Location(x=x, y=y, z=0),
                        capacity=2,
                        charging_rate=10.0,
                    )
                    registry.add_station(station)

            self._run_simulation(name, "Charging Stations", config, setup)

    # =========================================================================
    # 4. MOBILITY PATTERN TESTS
    # =========================================================================

    def _run_mobility_tests(self):
        """Test different user mobility patterns."""
        print("\n" + "=" * 50)
        print("4. USER MOBILITY PATTERNS")
        print("=" * 50)

        patterns = [
            ("Static Users", 0, "static"),
            ("Low Mobility (speed=1)", 1, "low"),
            ("Medium Mobility (speed=3)", 3, "medium"),
            ("High Mobility (speed=5)", 5, "high"),
            ("Clustered Static", 0, "clustered"),
        ]

        for name, speed, pattern in patterns:
            config = SimulationConfig(time_limit=500, user_count=15)
            config.uav.count = 3
            config.edge.count = 4

            def setup(_sim, spd=speed, pat=pattern):
                users = User.get_all()
                if pat == "clustered":
                    edges = EdgeServer.get_all()
                    for i, user in enumerate(users):
                        if edges:
                            edge = edges[i % len(edges)]
                            offset_x = np.random.uniform(-30, 30)
                            offset_y = np.random.uniform(-30, 30)
                            user.location = Location(
                                x=edge.location.x + offset_x, y=edge.location.y + offset_y, z=0
                            )
                else:
                    for user in users:
                        user.speed = spd

            self._run_simulation(name, "Mobility Patterns", config, setup)

    # =========================================================================
    # 5. SCHEDULING ALGORITHM TESTS
    # =========================================================================

    def _run_scheduling_tests(self):
        """Test different scheduling algorithms."""
        print("\n" + "=" * 50)
        print("5. SCHEDULING ALGORITHMS")
        print("=" * 50)

        algorithms = [
            ("Default (Load Balance)", None),
            ("Energy-First", SchedulingStrategy.ENERGY_FIRST),
            ("Latency-First", SchedulingStrategy.LATENCY_FIRST),
            ("Balanced", SchedulingStrategy.BALANCED),
            ("Utilization-Based", SchedulingStrategy.UTILIZATION),
        ]

        for name, strategy_enum in algorithms:
            config = SimulationConfig(time_limit=500, user_count=20)
            config.uav.count = 4
            config.edge.count = 4

            extra = {"strategy": name}

            # Note: Strategy must be set in the scheduler, which is inside simulation
            # We need a setup callback to change the scheduler strategy
            def setup(sim, strat=strategy_enum):
                if strat:
                    # Assuming we can replace or re-configure scheduling
                    # The simulator initializes EnergyAwareScheduler by default
                    # We might need to inject the strategy preference
                    sim.scheduler.strategy = strat

            result = self._run_simulation(name, "Scheduling", config, setup)
            if result.extra_metrics is None:
                result.extra_metrics = {}
            result.extra_metrics.update(extra)

    def save_results(self):
        """Save results to JSON."""
        json_path = self.output_dir / "benchmark_report.json"

        # Convert objects to dicts, flattening extra_metrics if needed
        data = []
        for r in self.results:
            d = r.to_dict()
            if r.extra_metrics:
                d.update(r.extra_metrics)
            data.append(d)

        with json_path.open("w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {json_path}")
        return json_path


def main():
    """Run unified benchmarks."""
    benchmark = UnifiedBenchmark()
    benchmark.run_all()
    benchmark.save_results()


if __name__ == "__main__":
    main()
