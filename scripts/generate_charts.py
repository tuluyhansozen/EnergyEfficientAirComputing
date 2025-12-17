"""Generate visualizations for benchmark reports."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend
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


def generate_basic_benchmark_charts(results_dir: Path):
    """Generate charts for basic benchmark report."""
    print("\n📊 Generating Basic Benchmark Charts...")

    json_path = results_dir / "benchmark_report.json"
    if not json_path.exists():
        print(f"  ⚠ JSON file not found: {json_path}")
        return

    data = load_benchmark_data(json_path)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # 1. Success Rate Bar Chart
    create_bar_comparison(
        data,
        "success_rate",
        "Task Success Rate by Configuration",
        str(charts_dir / "basic_success_rate.png"),
        "RdYlGn",
    )

    # 2. Energy Consumption Bar Chart
    create_bar_comparison(
        data,
        "total_energy",
        "Total Energy Consumption by Configuration",
        str(charts_dir / "basic_energy.png"),
        "YlOrRd",
    )

    # 3. Multi-metric Comparison
    create_multi_metric_chart(
        data,
        [("success_rate", "Success Rate"), ("avg_qos", "Avg QoS"), ("total_tasks", "Tasks")],
        "Performance Metrics Comparison",
        str(charts_dir / "basic_metrics.png"),
    )

    # 4. Energy vs Latency Scatter
    create_scatter_plot(
        data,
        "total_energy",
        "avg_latency",
        "Energy vs Latency Trade-off",
        str(charts_dir / "basic_tradeoff.png"),
    )

    # 5. Radar Chart
    create_radar_chart(
        data,
        [
            ("success_rate", "Success"),
            ("avg_qos", "QoS"),
            ("total_tasks", "Throughput"),
            ("total_energy", "Energy"),
            ("avg_latency", "Latency"),
        ],
        "Configuration Comparison Radar",
        str(charts_dir / "basic_radar.png"),
    )


def generate_advanced_benchmark_charts(results_dir: Path):
    """Generate charts for advanced benchmark report."""
    print("\n📊 Generating Advanced Benchmark Charts...")

    json_path = results_dir / "advanced_benchmark_report.json"
    if not json_path.exists():
        print(f"  ⚠ JSON file not found: {json_path}")
        return

    data = load_benchmark_data(json_path)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Group by category
    categories = {}
    for d in data:
        cat = d.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(d)

    # Generate charts for each category
    for cat_name, cat_data in categories.items():
        safe_name = cat_name.lower().replace(" ", "_")

        # Skip empty categories
        if not any(d.get("total_tasks", 0) > 0 for d in cat_data):
            continue

        # Bar chart for success rate
        create_bar_comparison(
            cat_data,
            "success_rate",
            f"{cat_name}: Success Rate Comparison",
            str(charts_dir / f"adv_{safe_name}_success.png"),
            "RdYlGn",
        )

        # Energy comparison
        create_bar_comparison(
            cat_data,
            "total_energy",
            f"{cat_name}: Energy Consumption",
            str(charts_dir / f"adv_{safe_name}_energy.png"),
            "YlOrRd",
        )

    # Overall heatmap
    valid_data = [d for d in data if d.get("total_tasks", 0) > 0]
    if valid_data:
        create_heatmap(
            valid_data,
            [
                ("success_rate", "Success"),
                ("avg_qos", "QoS"),
                ("total_energy", "Energy"),
                ("avg_latency", "Latency"),
            ],
            "Advanced Benchmark Heatmap",
            str(charts_dir / "adv_heatmap.png"),
        )

        # Overall radar
        create_radar_chart(
            valid_data[:8],  # Top 8
            [
                ("success_rate", "Success"),
                ("avg_qos", "QoS"),
                ("total_tasks", "Throughput"),
                ("avg_latency", "Latency"),
            ],
            "Top Configurations Comparison",
            str(charts_dir / "adv_radar.png"),
        )





def update_report_with_charts(report_path: Path, charts: list):
    """Add chart references to markdown report."""
    with report_path.open() as f:
        content = f.read()

    # Add charts section
    charts_section = "\n\n## Visualizations\n\n"
    for chart_name, chart_path, description in charts:
        rel_path = f"charts/{chart_path.name}"
        charts_section += f"### {chart_name}\n\n"
        charts_section += f"![{chart_name}]({rel_path})\n\n"
        charts_section += f"*{description}*\n\n"

    # Insert before conclusion/summary
    if "## Conclusion" in content:
        content = content.replace("## Conclusion", charts_section + "## Conclusion")
    elif "## Overall Summary" in content:
        content = content.replace("## Overall Summary", charts_section + "## Overall Summary")
    else:
        content += charts_section

    with report_path.open("w") as f:
        f.write(content)

    print(f"  Updated: {report_path}")


def main():
    """Generate all visualizations."""
    results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print("Results directory not found!")
        return

    print("=" * 60)
    print("AirCompSim Visualization Generator")
    print("=" * 60)

    # Generate charts
    generate_basic_benchmark_charts(results_dir)
    generate_advanced_benchmark_charts(results_dir)

    # Update reports with chart references
    charts_dir = results_dir / "charts"

    # Basic report charts
    basic_charts = [
        (
            "Success Rate Comparison",
            charts_dir / "basic_success_rate.png",
            "Task success rate across different infrastructure configurations.",
        ),
        (
            "Energy Consumption",
            charts_dir / "basic_energy.png",
            "Total energy consumed by each configuration.",
        ),
        (
            "Multi-Metric Comparison",
            charts_dir / "basic_metrics.png",
            "Normalized comparison of success rate, QoS, and throughput.",
        ),
        (
            "Energy-Latency Trade-off",
            charts_dir / "basic_tradeoff.png",
            "Scatter plot showing the relationship between energy and latency.",
        ),
        (
            "Configuration Radar",
            charts_dir / "basic_radar.png",
            "Radar chart comparing top configurations across all metrics.",
        ),
    ]

    basic_report = results_dir / "benchmark_report.md"
    if basic_report.exists():
        update_report_with_charts(basic_report, basic_charts)

    # Advanced report charts
    adv_charts = [
        (
            "UAV Positioning Success Rates",
            charts_dir / "adv_uav_positioning_success.png",
            "Comparison of UAV positioning strategies.",
        ),
        (
            "Mobility Pattern Impact",
            charts_dir / "adv_mobility_patterns_success.png",
            "Success rates under different user mobility patterns.",
        ),
        (
            "Scheduling Algorithm Comparison",
            charts_dir / "adv_scheduling_success.png",
            "Performance of different scheduling algorithms.",
        ),
        (
            "Performance Heatmap",
            charts_dir / "adv_heatmap.png",
            "Heatmap showing normalized metrics across all configurations.",
        ),
        (
            "Top Configurations Radar",
            charts_dir / "adv_radar.png",
            "Radar chart comparing the best performing configurations.",
        ),
    ]

    adv_report = results_dir / "advanced_benchmark_report.md"
    if adv_report.exists():
        update_report_with_charts(adv_report, adv_charts)

    print("\n" + "=" * 60)
    print("✅ Visualization generation complete!")
    print(f"   Charts saved to: {charts_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
