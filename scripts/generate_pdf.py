"""Generate combined PDF report from benchmark results."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_title_page(pdf: PdfPages, title: str, subtitle: str):
    """Create a title page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")

    # Title
    fig.text(
        0.5, 0.6, title, fontsize=32, ha="center", va="center", color="white", fontweight="bold"
    )

    # Subtitle
    fig.text(0.5, 0.45, subtitle, fontsize=16, ha="center", va="center", color="#888888")

    # Date
    date_str = datetime.now().strftime("%B %d, %Y")
    fig.text(
        0.5, 0.25, f"Generated: {date_str}", fontsize=12, ha="center", va="center", color="#666666"
    )

    # Footer
    fig.text(
        0.5,
        0.1,
        "Energy-Efficient Air Computing Simulator",
        fontsize=10,
        ha="center",
        va="center",
        color="#444444",
    )

    pdf.savefig(fig)
    plt.close(fig)


def create_section_header(pdf: PdfPages, title: str, description: str = ""):
    """Create a section header page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#16213e")

    fig.text(
        0.5, 0.55, title, fontsize=28, ha="center", va="center", color="white", fontweight="bold"
    )

    if description:
        fig.text(
            0.5,
            0.40,
            description,
            fontsize=14,
            ha="center",
            va="center",
            color="#aaaaaa",
            wrap=True,
        )

    pdf.savefig(fig)
    plt.close(fig)


def create_table_page(pdf: PdfPages, title: str, data: list, columns: list):
    """Create a page with a data table."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Title
    fig.text(0.5, 0.95, title, fontsize=18, ha="center", va="top", fontweight="bold")

    # Prepare table data
    # Pagination might be needed for large datasets, but we'll truncation/sample for now
    display_data = data[:25] # Show first 25 rows max
    
    table_data = []
    for d in display_data:
        row = []
        for col in columns:
            val = d.get(col["key"], "N/A")
            if col.get("format") == "percent":
                row.append(f"{val:.1%}" if isinstance(val, (int, float)) else str(val))
            elif col.get("format") == "float":
                row.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
            else:
                row.append(str(val))
        table_data.append(row)

    col_labels = [c["label"] for c in columns]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[1.0 / len(columns)] * len(columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Style header
    for i, _ in enumerate(col_labels):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")
            else:
                table[(i, j)].set_facecolor("#ffffff")
        
    if len(data) > 25:
         fig.text(0.5, 0.05, f"*Showing first 25 of {len(data)} rows", fontsize=10, ha="center", style="italic")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_image_page(pdf: PdfPages, image_path: Path, title: str = "", caption: str = ""):
    """Create a page with an embedded image."""
    fig = plt.figure(figsize=(11, 8.5))

    if title:
        fig.text(0.5, 0.96, title, fontsize=16, ha="center", va="top", fontweight="bold")

    if not image_path.exists():
        fig.text(0.5, 0.5, f"Image not found: {image_path.name}", ha="center", color="red")
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Load and display image
    img = mpimg.imread(str(image_path))
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    ax.imshow(img)
    ax.axis("off")

    if caption:
        fig.text(
            0.5,
            0.03,
            caption,
            fontsize=10,
            ha="center",
            va="bottom",
            style="italic",
            color="#666666",
        )

    pdf.savefig(fig)
    plt.close(fig)


def create_summary_page(pdf: PdfPages, paper_data: list, adv_data: list):
    """Create executive summary page."""
    fig = plt.figure(figsize=(11, 8.5))

    fig.text(0.5, 0.95, "Executive Summary", fontsize=24, ha="center", va="top", fontweight="bold")

    # Key findings
    findings = []
    findings.append("Overview:")
    findings.append(f"• Total scenarios run: {len(paper_data) + len(adv_data)}")
    findings.append(f"• Paper Replication Scenarios: {len(paper_data)}")
    findings.append(f"• Advanced Scenarios: {len(adv_data)}")
    findings.append("")

    # Paper replication findings
    valid_paper = [d for d in paper_data if d.get("total_tasks", 0) > 0]
    if valid_paper:
        best = max(valid_paper, key=lambda x: x.get("success_rate", 0))
        name = best.get("name", "Unknown")
        findings.append("Paper Replication Results:")
        findings.append(f"• Best Configuration: {name}")
        findings.append(f"  Success Rate: {best.get('success_rate', 0):.1%}")

    # Advanced benchmark findings
    valid_adv = [d for d in adv_data if d.get("total_tasks", 0) > 0]
    if valid_adv:
        findings.append("")
        findings.append("Advanced Benchmarks Highlights:")
        # Group by category
        categories = {}
        for d in valid_adv:
            cat = d.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(d)

        for cat, items in categories.items():
            best = max(items, key=lambda x: x.get("success_rate", 0))
            name = best.get("name", "Unknown")
            findings.append(f"• {cat}: Best is {name} ({best.get('success_rate', 0):.1%})")

    # Display findings
    y_pos = 0.82
    for finding in findings:
        if finding.endswith(":"):
            fig.text(0.1, y_pos, finding, fontsize=14, va="top", fontweight="bold")
        else:
            fig.text(0.1, y_pos, finding, fontsize=12, va="top")
        y_pos -= 0.04

    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(results_dir: Path, output_path: Path):
    """Generate the combined PDF report."""
    print("Generating PDF report...")

    charts_dir = results_dir / "charts"

    # Load Unified JSON data
    data = []
    json_path = results_dir / "benchmark_report.json"
    if json_path.exists():
        with json_path.open() as f:
            data = json.load(f)

    # Split data
    paper_data = [d for d in data if d.get("category") == "Paper Replication"]
    adv_data = [d for d in data if d.get("category") != "Paper Replication"]

    with PdfPages(str(output_path)) as pdf:
        # Title page
        create_title_page(pdf, "AirCompSim", "Unified Benchmark Report")

        # Executive summary
        create_summary_page(pdf, paper_data, adv_data)

        # 1. Paper Replication Section
        create_section_header(
            pdf,
            "1. Paper Replication",
            "Replication of results from AirCompSim paper (Figs 4-6)",
        )
        
        # Table for Paper Replication
        if paper_data:
             create_table_page(
                pdf,
                "Replication Data",
                paper_data,  # Has 25 rows
                [
                    {"label": "Name", "key": "name"},
                    {"label": "Success", "key": "success_rate", "format": "percent"},
                    {"label": "Latency", "key": "avg_latency", "format": "float"},
                    {"label": "Energy (J)", "key": "total_energy", "format": "float"},
                ],
            )
        
        # Charts for Paper Replication
        create_image_page(pdf, charts_dir / "basic_success_rate.png", "Figure 4: Success Rate", "Success Rate vs Users (Grouped by UAVs)")
        create_image_page(pdf, charts_dir / "basic_latency.png", "Figure 5: Service Time", "Average Service Time vs Users")
        create_image_page(pdf, charts_dir / "basic_energy.png", "Figure 6: Energy Consumption", "Total Energy Consumption vs Users")

        # 2. Advanced Section
        create_section_header(
            pdf,
            "2. Advanced Scenarios",
            "UAV Positioning, Charging, Mobility, and Scheduling",
        )
        
        # For each category in advanced data
        categories = ["UAV Positioning", "Charging Stations", "Mobility Patterns", "Scheduling"]
        
        for cat in categories:
            cat_data = [d for d in adv_data if d.get("category") == cat]
            if not cat_data:
                continue
                
            create_section_header(pdf, cat)
            
            create_table_page(
                pdf,
                f"{cat} Results",
                cat_data,
                [
                    {"label": "Name", "key": "name"},
                    {"label": "Success", "key": "success_rate", "format": "percent"},
                    {"label": "Latency", "key": "avg_latency", "format": "float"},
                    {"label": "Energy", "key": "total_energy", "format": "float"},
                ],
            )
            
            # Add specific charts based on category knowledge
            if cat == "UAV Positioning":
                create_image_page(pdf, charts_dir / "adv_uav_positioning_success.png", "Success Rate")
                create_image_page(pdf, charts_dir / "adv_uav_positioning_energy.png", "Energy Consumption")
            elif cat == "Charging Stations":
                create_image_page(pdf, charts_dir / "adv_charging_stations_success.png", "Success Rate")
            elif cat == "Mobility Patterns":
                create_image_page(pdf, charts_dir / "adv_mobility_patterns_success.png", "Success Rate")
            elif cat == "Scheduling":
                create_image_page(pdf, charts_dir / "adv_scheduling_success.png", "Success Rate")
                create_image_page(pdf, charts_dir / "adv_scheduling_latency.png", "Latency")

    print(f"PDF Report saved to: {output_path}")


def main():
    results_dir = Path(__file__).parent.parent / "results"
    output_path = results_dir / "AirCompSim_Benchmark_Report.pdf"
    generate_pdf_report(results_dir, output_path)


if __name__ == "__main__":
    main()
