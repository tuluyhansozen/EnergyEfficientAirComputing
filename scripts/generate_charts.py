"""Generate visualizations for benchmark reports."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend
from typing import Optional

import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def load_benchmark_data(json_path: Path) -> list:
    """Load benchmark results from JSON."""
    with json_path.open() as f:
        return json.load(f)


def create_bar_comparison(
    data: list, metric: str, title: str, output_path: str, color_scheme: str = "viridis"
):
    """Create bar chart comparing configurations."""
    names = [d.get("name", d.get("config_name", "Unknown")) for d in data]
    values = [d.get(metric, 0) for d in data]

    _, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.get_cmap(color_scheme)(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1%}" if metric == "success_rate" else f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Configuration")
    ax.set_ylabel(metric.replace("_", " ").title())

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def create_multi_metric_chart(data: list, metrics: list, title: str, output_path: str):
    """Create grouped bar chart for multiple metrics."""
    names = [
        d.get("name", d.get("config_name", "Unknown"))[:15] for d in data
    ]  # Truncate long names
    x = np.arange(len(names))
    width = 0.8 / len(metrics)

    _, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

    for i, (metric, label) in enumerate(metrics):
        values = [d.get(metric, 0) for d in data]
        # Normalize for comparison
        normalized = [v / max(values) * 100 for v in values] if max(values) > 0 else values

        offset = width * (i - len(metrics) / 2 + 0.5)
        ax.bar(x + offset, normalized, width, label=label, color=colors[i])

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Normalized Value (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def create_scatter_plot(data: list, x_metric: str, y_metric: str, title: str, output_path: str):
    """Create scatter plot of two metrics."""
    _, ax = plt.subplots(figsize=(10, 8))

    x_values = [d.get(x_metric, 0) for d in data]
    y_values = [d.get(y_metric, 0) for d in data]
    names = [d.get("name", d.get("config_name", "Unknown")) for d in data]

    # Color by success rate
    colors = [d.get("success_rate", 0) for d in data]

    scatter = ax.scatter(
        x_values,
        y_values,
        c=colors,
        cmap="RdYlGn",
        s=150,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Add labels
    for i, name in enumerate(names):
        ax.annotate(
            name[:12],
            (x_values[i], y_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Success Rate")

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def create_radar_chart(data: list, metrics: list, title: str, output_path: str):
    """Create radar chart comparing configurations."""
    # Select top 5 configurations
    selected = data[:5] if len(data) > 5 else data

    categories = [m[1] for m in metrics]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    _, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

    colors = plt.cm.Set2(np.linspace(0, 1, len(selected)))

    for i, d in enumerate(selected):
        values = []
        for metric, _ in metrics:
            val = d.get(metric, 0)
            # Normalize each metric
            max_val = max(x.get(metric, 1) for x in data) or 1
            values.append(val / max_val * 100)
        values += values[:1]

        name = d.get("name", d.get("config_name", "Unknown"))
        ax.plot(angles, values, "o-", linewidth=2, label=name[:15], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, fontweight="bold", pad=20, size=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def create_heatmap(data: list, metrics: list, title: str, output_path: str):
    """Create heatmap of metrics across configurations."""
    names = [d.get("name", d.get("config_name", "Unknown"))[:18] for d in data]
    metric_names = [m[1] for m in metrics]

    # Build matrix
    matrix = []
    for d in data:
        row = []
        for metric, _ in metrics:
            val = d.get(metric, 0)
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)

    # Normalize each column
    for j in range(matrix.shape[1]):
        col_max = matrix[:, j].max() or 1
        matrix[:, j] = matrix[:, j] / col_max * 100

    _, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels(names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Score (%)", rotation=-90, va="bottom")

    # Add value annotations
    for i in range(len(names)):
        for j in range(len(metric_names)):
            ax.text(
                j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black", fontsize=9
            )

    ax.set_title(title, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def create_line_comparison(
    data: list,
    x_metric: str,
    y_metric: str,
    group_by: str,
    title: str,
    output_path: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    """Create line chart comparing metrics grouped by a category."""
    plt.figure(figsize=(10, 6))

    # Identify groups
    groups = sorted({d[group_by] for d in data})

    for group in groups:
        # Filter data for this group
        group_data = [d for d in data if d[group_by] == group]
        # Sort by x metric
        group_data.sort(key=lambda x: x[x_metric])

        x = [d[x_metric] for d in group_data]
        y = [d[y_metric] for d in group_data]

        plt.plot(x, y, marker="o", label=f"{group} {group_by.upper()}")

    plt.title(title, fontweight="bold", pad=15)
    plt.xlabel(x_label or x_metric.replace("_", " ").title())
    plt.ylabel(y_label or y_metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set y-axis limits for percentages if applicable
    if "rate" in y_metric or "percentage" in y_metric:
        plt.ylim(-0.05, 1.05)
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def generate_paper_replication_charts(data: list, charts_dir: Path):
    """Generate charts for Paper Replication section (Figs 4-6)."""
    print("  Generating Paper Replication Charts...")

    # Filter for category
    subset = [d for d in data if d.get("category") == "Paper Replication"]
    if not subset:
        print("    ⚠ No data found for Paper Replication")
        return

    # Figure 4: Success Rate vs Users (Grouped by UAVs)
    create_line_comparison(
        subset,
        x_metric="users",
        y_metric="success_rate",
        group_by="uavs",
        title="Avg Task Success Rate (Fig. 4)",
        output_path=str(charts_dir / "basic_success_rate.png"),
        x_label="Number of Users",
        y_label="Success Rate",
    )

    # Figure 5: Service Time (Latency) vs Users
    create_line_comparison(
        subset,
        x_metric="users",
        y_metric="avg_latency",
        group_by="uavs",
        title="Avg Service Time (Fig. 5)",
        output_path=str(charts_dir / "basic_latency.png"),
        x_label="Number of Users",
        y_label="Avg Service Time (s)",
    )

    # Figure 6: Energy/Utilization
    create_line_comparison(
        subset,
        x_metric="users",
        y_metric="total_energy",
        group_by="uavs",
        title="Total Energy Consumption",
        output_path=str(charts_dir / "basic_energy.png"),
        x_label="Number of Users",
        y_label="Total Energy (J)",
    )


def generate_advanced_charts(data: list, charts_dir: Path):
    """Generate charts for advanced scenarios."""
    print("  Generating Advanced Scenario Charts...")

    # 1. UAV Positioning
    subset = [d for d in data if d.get("category") == "UAV Positioning"]
    if subset:
        create_bar_comparison(
            subset,
            "success_rate",
            "UAV Positioning - Success Rate",
            str(charts_dir / "adv_uav_positioning_success.png"),
        )
        create_bar_comparison(
            subset,
            "total_energy",
            "UAV Positioning - Energy",
            str(charts_dir / "adv_uav_positioning_energy.png"),
            "YlOrRd",
        )

    # 2. Charging Stations
    subset = [d for d in data if d.get("category") == "Charging Stations"]
    if subset:
        create_bar_comparison(
            subset,
            "success_rate",
            "Charging Impact - Success Rate",
            str(charts_dir / "adv_charging_stations_success.png"),
        )

    # 3. Mobility Patterns
    subset = [d for d in data if d.get("category") == "Mobility Patterns"]
    if subset:
        create_bar_comparison(
            subset,
            "success_rate",
            "Mobility Impact - Success Rate",
            str(charts_dir / "adv_mobility_patterns_success.png"),
        )

    # 4. Scheduling
    subset = [d for d in data if d.get("category") == "Scheduling"]
    if subset:
        create_bar_comparison(
            subset,
            "success_rate",
            "Scheduling - Success Rate",
            str(charts_dir / "adv_scheduling_success.png"),
        )
        create_bar_comparison(
            subset,
            "avg_latency",
            "Scheduling - Latency",
            str(charts_dir / "adv_scheduling_latency.png"),
            "Blues",
        )

    # 5. Overall Radar (Top 5 Best Success Rates)
    valid_data = [d for d in data if d.get("total_tasks", 0) > 0]
    valid_data.sort(key=lambda x: x.get("success_rate", 0), reverse=True)
    top_5 = valid_data[:5]

    if top_5:
        create_radar_chart(
            top_5,
            [
                ("success_rate", "Success"),
                ("avg_qos", "QoS"),
                ("total_tasks", "Throughput"),
                ("total_energy", "Energy"),
                ("avg_latency", "Latency"),
            ],
            "Top 5 Configurations Radar",
            str(charts_dir / "adv_radar.png"),
        )


def main():
    """Generate all visualizations."""
    results_dir = Path(__file__).parent.parent / "results"
    json_path = results_dir / "benchmark_report.json"

    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    print("=" * 60)
    print("AirCompSim Visualization Generator")
    print("=" * 60)

    data = load_benchmark_data(json_path)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    generate_paper_replication_charts(data, charts_dir)
    generate_advanced_charts(data, charts_dir)

    print("\n" + "=" * 60)
    print("✅ Visualization generation complete!")
    print(f"   Charts saved to: {charts_dir}")


if __name__ == "__main__":
    main()
