"""CLI entry point for AirCompSim.

Run simulations from command line.
"""

import argparse
import logging
import sys
from pathlib import Path

from aircompsim import Simulation, SimulationConfig, __version__
from aircompsim.config.loader import load_config


def setup_logging(level: str) -> None:
    """Configure logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="aircompsim", description="Energy-Efficient Air Computing Simulator"
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file (YAML or JSON)"
    )

    parser.add_argument(
        "--time-limit", "-t", type=float, default=1000, help="Simulation time limit (default: 1000)"
    )

    parser.add_argument("--users", "-u", type=int, default=20, help="Number of users (default: 20)")

    parser.add_argument("--uavs", type=int, default=5, help="Number of UAVs (default: 5)")

    parser.add_argument("--edges", type=int, default=4, help="Number of edge servers (default: 4)")

    parser.add_argument("--drl", action="store_true", help="Enable DRL training")

    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of DRL episodes (default: 500)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--output", "-o", type=str, default="results", help="Output directory (default: results)"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"AirCompSim v{__version__}")

    # Load or create config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        config_data = load_config(config_path)
        config = SimulationConfig.from_dict(config_data)
    else:
        config = SimulationConfig()

    # Apply CLI overrides
    config.time_limit = args.time_limit
    config.user_count = args.users
    config.uav.count = args.uavs
    config.edge.count = args.edges
    config.drl.enabled = args.drl
    config.drl.episodes = args.episodes
    config.output_dir = args.output

    # Run simulation
    if args.drl:
        results = run_drl_training(config)
    else:
        results = run_single_simulation(config)

    # Print results
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"Total tasks:      {results.total_tasks}")
    print(f"Successful tasks: {results.successful_tasks}")
    print(f"Success rate:     {results.success_rate:.2%}")
    print(f"Average latency:  {results.avg_latency:.4f}s")
    print(f"Average QoS:      {results.avg_qos:.1f}")
    print(f"Total energy:     {results.total_energy:.2f}J")
    print("=" * 50)

    return 0


def run_single_simulation(config: SimulationConfig):
    """Run a single simulation."""
    sim = Simulation(config)
    sim.initialize()
    return sim.run()


def run_drl_training(config: SimulationConfig):
    """Run DRL training across multiple episodes."""
    from aircompsim.drl import DDQNAgent

    logger = logging.getLogger(__name__)
    logger.info(f"Starting DRL training for {config.drl.episodes} episodes")

    # Initialize agent
    state_size = config.uav.count * 3  # x, y, battery per UAV
    action_size = config.uav.count * 4  # 4 directions per UAV

    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=config.drl.learning_rate,
        discount_factor=config.drl.discount_factor,
    )

    best_results = None

    for episode in range(config.drl.episodes):
        sim = Simulation(config)
        sim.set_agent(agent)
        sim.initialize()
        results = sim.run()

        if best_results is None or results.success_rate > best_results.success_rate:
            best_results = results

        if (episode + 1) % 10 == 0:
            logger.info(
                f"Episode {episode + 1}/{config.drl.episodes}: "
                f"success_rate={results.success_rate:.2%}, "
                f"epsilon={agent.epsilon:.3f}"
            )

    # Save agent
    agent.save(f"{config.output_dir}/agent.pt")
    logger.info(f"Agent saved to {config.output_dir}/agent.pt")

    return best_results


if __name__ == "__main__":
    sys.exit(main())
