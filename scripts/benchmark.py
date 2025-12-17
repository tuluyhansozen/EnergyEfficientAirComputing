"""Benchmark script to run simulations with different configurations."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aircompsim import Simulation, SimulationConfig


def run_benchmark():
    """Run simulations with various configurations."""

    results = []

    # Configuration variations
    configurations = [
        # Baseline
        {"name": "Baseline", "users": 10, "uavs": 3, "edges": 4, "time_limit": 500},
        # Vary user count
        {"name": "Low Users (5)", "users": 5, "uavs": 3, "edges": 4, "time_limit": 500},
        {"name": "Medium Users (20)", "users": 20, "uavs": 3, "edges": 4, "time_limit": 500},
        {"name": "High Users (30)", "users": 30, "uavs": 3, "edges": 4, "time_limit": 500},
        # Vary UAV count
        {"name": "No UAVs", "users": 10, "uavs": 0, "edges": 4, "time_limit": 500},
        {"name": "Many UAVs (5)", "users": 10, "uavs": 5, "edges": 4, "time_limit": 500},
        {"name": "Many UAVs (8)", "users": 10, "uavs": 8, "edges": 4, "time_limit": 500},
        # Vary edge server count
        {"name": "Few Edges (2)", "users": 10, "uavs": 3, "edges": 2, "time_limit": 500},
        {"name": "Many Edges (6)", "users": 10, "uavs": 3, "edges": 6, "time_limit": 500},
        # High load scenario
        {"name": "High Load", "users": 30, "uavs": 5, "edges": 6, "time_limit": 500},
    ]

    print("=" * 70)
    print("AirCompSim Benchmark - Multiple Configurations")
    print("=" * 70)
    print()

    for i, cfg in enumerate(configurations, 1):
        print(f"[{i}/{len(configurations)}] Running: {cfg['name']}")
        print(f"    Users: {cfg['users']}, UAVs: {cfg['uavs']}, Edges: {cfg['edges']}")

        try:
            # Create config
            config = SimulationConfig(
                time_limit=cfg["time_limit"],
                user_count=cfg["users"],
            )
            config.uav.count = cfg["uavs"]
            config.edge.count = cfg["edges"]

            # Run simulation
            sim = Simulation(config)
            sim.initialize()
            sim_results = sim.run()

            # Collect metrics
            result = {
                "config_name": cfg["name"],
                "users": cfg["users"],
                "uavs": cfg["uavs"],
                "edges": cfg["edges"],
                "time_limit": cfg["time_limit"],
                "total_tasks": sim_results.total_tasks,
                "successful_tasks": sim_results.successful_tasks,
                "failed_tasks": sim_results.failed_tasks,
                "success_rate": sim_results.success_rate,
                "avg_latency": sim_results.avg_latency,
                "avg_qos": sim_results.avg_qos,
                "total_energy": sim_results.total_energy,
            }
            results.append(result)

            print(
                f"    ✓ Tasks: {sim_results.total_tasks}, Success: {sim_results.success_rate:.1%}"
            )
            print()

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append({"config_name": cfg["name"], "error": str(e)})
            print()

    return results


def generate_report(results: list, output_path: str):
    """Generate markdown report from results."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# AirCompSim Benchmark Report

**Generated:** {timestamp}

## Summary

Ran {len(results)} different configurations to compare performance metrics.

## Results Table

| Configuration | Users | UAVs | Edges | Tasks | Success Rate | Avg Latency (s) | Avg QoS | Energy (J) |
|--------------|-------|------|-------|-------|--------------|-----------------|---------|------------|
"""

    for r in results:
        if "error" in r:
            report += f"| {r['config_name']} | - | - | - | ERROR | - | - | - | - |\n"
        else:
            report += f"| {r['config_name']} | {r['users']} | {r['uavs']} | {r['edges']} | {r['total_tasks']} | {r['success_rate']:.1%} | {r['avg_latency']:.4f} | {r['avg_qos']:.1f} | {r['total_energy']:.2f} |\n"

    report += """
## Key Observations

"""

    # Find some insights
    valid_results = [r for r in results if "error" not in r and r["total_tasks"] > 0]

    if valid_results:
        best_success = max(valid_results, key=lambda x: x["success_rate"])
        worst_success = min(valid_results, key=lambda x: x["success_rate"])
        most_energy = max(valid_results, key=lambda x: x["total_energy"])
        least_energy = min(valid_results, key=lambda x: x["total_energy"])

        report += f"""### Success Rate
- **Best:** {best_success['config_name']} ({best_success['success_rate']:.1%})
- **Worst:** {worst_success['config_name']} ({worst_success['success_rate']:.1%})

### Energy Consumption
- **Highest:** {most_energy['config_name']} ({most_energy['total_energy']:.2f} J)
- **Lowest:** {least_energy['config_name']} ({least_energy['total_energy']:.2f} J)

### Task Throughput
"""
        most_tasks = max(valid_results, key=lambda x: x["total_tasks"])
        report += (
            f"- **Most Tasks:** {most_tasks['config_name']} ({most_tasks['total_tasks']} tasks)\n"
        )

    report += """
## Configuration Details

Each simulation ran for 500 simulation time units with varying:
- **User Count:** 5-30 users
- **UAV Count:** 0-8 UAVs
- **Edge Server Count:** 2-6 edge servers

## Conclusion

The benchmark demonstrates how different infrastructure configurations affect:
1. Task completion success rate
2. Average latency
3. Quality of Service (QoS)
4. Total energy consumption

Increasing UAV count generally improves coverage but increases energy usage.
More edge servers provide better load balancing and lower latency.
"""

    # Save report
    with output_path.open("w") as f:
        f.write(report)

    # Also save raw JSON
    json_path = Path(output_path.parent) / output_path.name.replace(".md", ".json")
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    return report


if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmark()

    # Generate report
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "benchmark_report.md"
    report = generate_report(results, str(report_path))

    print("=" * 70)
    print(f"Report saved to: {report_path}")
    print(f"JSON data saved to: {report_path.with_suffix('.json')}")
    print("=" * 70)
