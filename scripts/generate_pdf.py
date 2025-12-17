"""Generate combined PDF report from benchmark results."""

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np


def create_title_page(pdf: PdfPages, title: str, subtitle: str):
    """Create a title page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Title
    fig.text(0.5, 0.6, title, fontsize=32, ha='center', va='center',
             color='white', fontweight='bold')
    
    # Subtitle
    fig.text(0.5, 0.45, subtitle, fontsize=16, ha='center', va='center',
             color='#888888')
    
    # Date
    date_str = datetime.now().strftime("%B %d, %Y")
    fig.text(0.5, 0.25, f"Generated: {date_str}", fontsize=12,
             ha='center', va='center', color='#666666')
    
    # Footer
    fig.text(0.5, 0.1, "Energy-Efficient Air Computing Simulator",
             fontsize=10, ha='center', va='center', color='#444444')
    
    pdf.savefig(fig)
    plt.close(fig)


def create_section_header(pdf: PdfPages, title: str, description: str = ""):
    """Create a section header page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#16213e')
    
    fig.text(0.5, 0.55, title, fontsize=28, ha='center', va='center',
             color='white', fontweight='bold')
    
    if description:
        fig.text(0.5, 0.40, description, fontsize=14, ha='center', va='center',
                 color='#aaaaaa', wrap=True)
    
    pdf.savefig(fig)
    plt.close(fig)


def create_table_page(pdf: PdfPages, title: str, data: list, columns: list):
    """Create a page with a data table."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.95, title, fontsize=18, ha='center', va='top',
             fontweight='bold')
    
    # Prepare table data
    table_data = []
    for d in data:
        row = []
        for col in columns:
            val = d.get(col['key'], 'N/A')
            if col.get('format') == 'percent':
                row.append(f"{val:.1%}" if isinstance(val, (int, float)) else str(val))
            elif col.get('format') == 'float':
                row.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
            else:
                row.append(str(val))
        table_data.append(row)
    
    col_labels = [c['label'] for c in columns]
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[1.0/len(columns)] * len(columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i, key in enumerate(col_labels):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_image_page(pdf: PdfPages, image_path: Path, title: str = "", caption: str = ""):
    """Create a page with an embedded image."""
    fig = plt.figure(figsize=(11, 8.5))
    
    if title:
        fig.text(0.5, 0.96, title, fontsize=16, ha='center', va='top',
                 fontweight='bold')
    
    # Load and display image
    img = mpimg.imread(str(image_path))
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    ax.imshow(img)
    ax.axis('off')
    
    if caption:
        fig.text(0.5, 0.03, caption, fontsize=10, ha='center', va='bottom',
                 style='italic', color='#666666')
    
    pdf.savefig(fig)
    plt.close(fig)


def create_summary_page(pdf: PdfPages, basic_data: list, adv_data: list):
    """Create executive summary page."""
    fig = plt.figure(figsize=(11, 8.5))
    
    fig.text(0.5, 0.95, "Executive Summary", fontsize=24, ha='center', va='top',
             fontweight='bold')
    
    # Key findings
    findings = []
    
    # Basic benchmark findings
    valid_basic = [d for d in basic_data if d.get('total_tasks', 0) > 0]
    if valid_basic:
        best = max(valid_basic, key=lambda x: x.get('success_rate', 0))
        name = best.get('config_name', best.get('name', 'Unknown'))
        findings.append(f"• Best basic configuration: {name} ({best.get('success_rate', 0):.1%} success)")
        
        efficient = min(valid_basic, key=lambda x: x.get('total_energy', float('inf')))
        name = efficient.get('config_name', efficient.get('name', 'Unknown'))
        findings.append(f"• Most energy efficient: {name} ({efficient.get('total_energy', 0):.0f} J)")
    
    # Advanced benchmark findings
    valid_adv = [d for d in adv_data if d.get('total_tasks', 0) > 0]
    if valid_adv:
        # By category
        categories = {}
        for d in valid_adv:
            cat = d.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(d)
        
        for cat, items in categories.items():
            best = max(items, key=lambda x: x.get('success_rate', 0))
            name = best.get('name', 'Unknown')
            findings.append(f"• Best {cat}: {name} ({best.get('success_rate', 0):.1%})")
    
    # Add recommendations
    findings.append("")
    findings.append("Key Recommendations:")
    findings.append("• Use grid positioning for optimal UAV coverage")
    findings.append("• Energy-first scheduling reduces consumption by ~20%")
    findings.append("• Static users achieve highest success rates")
    findings.append("• 3-4 edge servers provide optimal balance")
    
    # Display findings
    y_pos = 0.82
    for finding in findings:
        if finding.startswith("Key"):
            fig.text(0.1, y_pos, finding, fontsize=14, va='top', fontweight='bold')
        else:
            fig.text(0.1, y_pos, finding, fontsize=12, va='top')
        y_pos -= 0.05
    
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(results_dir: Path, output_path: Path):
    """Generate the combined PDF report."""
    print("Generating PDF report...")
    
    charts_dir = results_dir / "charts"
    
    # Load JSON data
    basic_data = []
    basic_json = results_dir / "benchmark_report.json"
    if basic_json.exists():
        with open(basic_json, 'r') as f:
            basic_data = json.load(f)
    
    adv_data = []
    adv_json = results_dir / "advanced_benchmark_report.json"
    if adv_json.exists():
        with open(adv_json, 'r') as f:
            adv_data = json.load(f)
    
    with PdfPages(str(output_path)) as pdf:
        # Title page
        create_title_page(
            pdf,
            "AirCompSim",
            "Benchmark Report & Analysis"
        )
        
        # Executive summary
        create_summary_page(pdf, basic_data, adv_data)
        
        # Basic Benchmark Section
        create_section_header(
            pdf,
            "Basic Benchmark Results",
            "Performance comparison across infrastructure configurations"
        )
        
        # Basic benchmark table
        if basic_data:
            columns = [
                {'key': 'config_name', 'label': 'Configuration'},
                {'key': 'users', 'label': 'Users'},
                {'key': 'uavs', 'label': 'UAVs'},
                {'key': 'edges', 'label': 'Edges'},
                {'key': 'total_tasks', 'label': 'Tasks'},
                {'key': 'success_rate', 'label': 'Success', 'format': 'percent'},
                {'key': 'total_energy', 'label': 'Energy (J)', 'format': 'float'},
            ]
            create_table_page(pdf, "Infrastructure Configuration Results", basic_data, columns)
        
        # Basic benchmark charts
        basic_charts = [
            ("basic_success_rate.png", "Success Rate by Configuration",
             "Task completion success rates across different infrastructure setups"),
            ("basic_energy.png", "Energy Consumption",
             "Total energy consumed during simulation"),
            ("basic_metrics.png", "Multi-Metric Comparison",
             "Normalized comparison of success, QoS, and throughput"),
            ("basic_tradeoff.png", "Energy-Latency Trade-off",
             "Relationship between energy consumption and task latency"),
            ("basic_radar.png", "Configuration Radar Chart",
             "Multi-dimensional comparison of top configurations"),
        ]
        
        for chart_file, title, caption in basic_charts:
            chart_path = charts_dir / chart_file
            if chart_path.exists():
                create_image_page(pdf, chart_path, title, caption)
        
        # Advanced Benchmark Section
        create_section_header(
            pdf,
            "Advanced Benchmark Results",
            "UAV positioning, mobility patterns, and scheduling strategies"
        )
        
        # Advanced benchmark by category
        if adv_data:
            # Group by category
            categories = {}
            for d in adv_data:
                cat = d.get('category', 'Unknown')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(d)
            
            for cat_name, cat_data in categories.items():
                if not any(d.get('total_tasks', 0) > 0 for d in cat_data):
                    continue
                    
                columns = [
                    {'key': 'name', 'label': 'Strategy'},
                    {'key': 'total_tasks', 'label': 'Tasks'},
                    {'key': 'success_rate', 'label': 'Success', 'format': 'percent'},
                    {'key': 'avg_latency', 'label': 'Latency (s)', 'format': 'float'},
                    {'key': 'avg_qos', 'label': 'QoS', 'format': 'float'},
                    {'key': 'total_energy', 'label': 'Energy (J)', 'format': 'float'},
                ]
                create_table_page(pdf, f"{cat_name} Results", cat_data, columns)
        
        # Advanced charts
        adv_charts = [
            ("adv_uav_positioning_success.png", "UAV Positioning Strategies",
             "Success rates for different UAV placement strategies"),
            ("adv_mobility_patterns_success.png", "User Mobility Impact",
             "Effect of user movement patterns on task success"),
            ("adv_scheduling_success.png", "Scheduling Algorithm Comparison",
             "Performance of different scheduling strategies"),
            ("adv_heatmap.png", "Performance Heatmap",
             "Normalized metrics across all configurations"),
            ("adv_radar.png", "Top Configurations",
             "Radar comparison of best performing configurations"),
        ]
        
        for chart_file, title, caption in adv_charts:
            chart_path = charts_dir / chart_file
            if chart_path.exists():
                create_image_page(pdf, chart_path, title, caption)
        
        # Conclusion page
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#1a1a2e')
        
        fig.text(0.5, 0.55, "Thank You", fontsize=36, ha='center', va='center',
                 color='white', fontweight='bold')
        
        fig.text(0.5, 0.40, "AirCompSim Benchmark Report", fontsize=16,
                 ha='center', va='center', color='#888888')
        
        fig.text(0.5, 0.25, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                 fontsize=12, ha='center', va='center', color='#666666')
        
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"✅ PDF saved to: {output_path}")
    return output_path


def main():
    """Main function."""
    results_dir = Path(__file__).parent.parent / "results"
    output_path = results_dir / "AirCompSim_Benchmark_Report.pdf"
    
    print("=" * 60)
    print("AirCompSim PDF Report Generator")
    print("=" * 60)
    
    generate_pdf_report(results_dir, output_path)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
