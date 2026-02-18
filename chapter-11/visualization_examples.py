"""
Chapter 11: Data Visualization Techniques
Section 11.2 — Key Tools and Libraries for Crafting Visual Representations

Demonstrates a variety of visualization types using Matplotlib:
- Bar charts (grouped and stacked)
- Line charts with trend analysis
- Pie/donut charts
- Heatmaps
- Scatter plots with annotations
- Subplots for dashboard-style layouts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def business_dashboard():
    """Create a multi-panel business analytics dashboard."""
    print("=" * 60)
    print("VISUALIZATION: Business Analytics Dashboard")
    print("=" * 60)

    np.random.seed(42)

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    years = ["2023", "2024"]
    revenue_2023 = [320, 380, 350, 420]
    revenue_2024 = [380, 440, 410, 490]
    costs_2024 = [280, 310, 295, 340]
    profit_2024 = [r - c for r, c in zip(revenue_2024, costs_2024)]

    departments = ["Engineering", "Marketing", "Sales", "Support", "Operations"]
    dept_budget = [45, 25, 20, 15, 12]

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_users = [1200, 1350, 1500, 1450, 1600, 1750,
                     1800, 1950, 2100, 2000, 2200, 2400]
    monthly_churn = [3.2, 2.8, 2.5, 2.9, 2.3, 2.1,
                     1.9, 2.0, 1.8, 2.2, 1.7, 1.5]

    regions = ["North", "South", "East", "West"]
    products = ["Product A", "Product B", "Product C"]
    sales_matrix = np.array([
        [85, 92, 78],
        [70, 88, 95],
        [90, 75, 82],
        [65, 80, 90],
    ])

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Business Analytics Dashboard — FY 2024", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Grouped Bar Chart (Revenue Comparison) ---
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(len(quarters))
    width = 0.35
    ax1.bar(x - width/2, revenue_2023, width, label="2023", color="#2B579A", alpha=0.8)
    ax1.bar(x + width/2, revenue_2024, width, label="2024", color="#C00000", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quarters)
    ax1.set_ylabel("Revenue ($K)")
    ax1.set_title("Quarterly Revenue: YoY", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Stacked Bar Chart (Revenue vs Costs) ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(quarters, costs_2024, label="Costs", color="#A5A5A5")
    ax2.bar(quarters, profit_2024, bottom=costs_2024, label="Profit", color="#2E8B57")
    ax2.set_ylabel("Amount ($K)")
    ax2.set_title("Revenue Breakdown: 2024", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Donut Chart (Budget by Department) ---
    ax3 = fig.add_subplot(2, 3, 3)
    colors = ["#2B579A", "#C00000", "#2E8B57", "#FF8C00", "#6A0DAD"]
    wedges, texts, autotexts = ax3.pie(
        dept_budget, labels=departments, autopct="%1.0f%%",
        colors=colors, startangle=90, pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor="white")
    )
    for text in autotexts:
        text.set_fontsize(8)
    ax3.set_title("Budget Allocation by Dept", fontweight="bold")

    # --- Panel 4: Dual-Axis Line Chart (Users + Churn) ---
    ax4 = fig.add_subplot(2, 3, 4)
    color1, color2 = "#2B579A", "#C00000"
    ax4.plot(months, monthly_users, "o-", color=color1, linewidth=2, markersize=5, label="Active Users")
    ax4.set_ylabel("Active Users", color=color1)
    ax4.tick_params(axis="y", labelcolor=color1)
    ax4.set_title("User Growth & Churn Rate", fontweight="bold")
    ax4.tick_params(axis="x", rotation=45)

    ax4b = ax4.twinx()
    ax4b.plot(months, monthly_churn, "s--", color=color2, linewidth=2, markersize=5, label="Churn %")
    ax4b.set_ylabel("Churn Rate (%)", color=color2)
    ax4b.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # --- Panel 5: Heatmap (Sales by Region × Product) ---
    ax5 = fig.add_subplot(2, 3, 5)
    im = ax5.imshow(sales_matrix, cmap="YlOrRd", aspect="auto")
    ax5.set_xticks(range(len(products)))
    ax5.set_yticks(range(len(regions)))
    ax5.set_xticklabels(products)
    ax5.set_yticklabels(regions)
    ax5.set_title("Sales Heatmap (Region × Product)", fontweight="bold")
    for i in range(len(regions)):
        for j in range(len(products)):
            ax5.text(j, i, f"{sales_matrix[i, j]}", ha="center", va="center",
                    color="white" if sales_matrix[i, j] > 85 else "black", fontsize=10)
    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # --- Panel 6: Scatter Plot with Annotations ---
    ax6 = fig.add_subplot(2, 3, 6)
    np.random.seed(42)
    ad_spend = np.random.uniform(10, 100, 20)
    conversions = 2.5 * ad_spend + np.random.normal(0, 20, 20)
    ax6.scatter(ad_spend, conversions, c="#2B579A", s=60, alpha=0.7, edgecolors="black", linewidths=0.5)
    # Trend line
    z = np.polyfit(ad_spend, conversions, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ad_spend.min(), ad_spend.max(), 100)
    ax6.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trend (slope={z[0]:.1f})")
    ax6.set_xlabel("Ad Spend ($K)")
    ax6.set_ylabel("Conversions")
    ax6.set_title("Ad Spend vs Conversions", fontweight="bold")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("sample-outputs/chapter-11/business_dashboard.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: sample-outputs/chapter-11/business_dashboard.png")
    print("  Contains 6 visualization types: grouped bar, stacked bar,")
    print("  donut chart, dual-axis line, heatmap, and annotated scatter.")


def storytelling_visualization():
    """Create a narrative-driven visualization with annotations."""
    print("\n\n" + "=" * 60)
    print("VISUALIZATION: Data Storytelling Example")
    print("=" * 60)
    print("Telling the story of a company's growth journey.\n")

    months = np.arange(1, 25)
    revenue = np.cumsum(np.random.normal(15, 5, 24)) + 100
    revenue = np.clip(revenue, 50, None)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(months, revenue, "-", color="#2B579A", linewidth=2.5)
    ax.fill_between(months, revenue, alpha=0.1, color="#2B579A")

    # Annotate key events
    events = {
        3: ("Product Launch", revenue[2]),
        8: ("Series A Funding", revenue[7]),
        14: ("International Expansion", revenue[13]),
        20: ("1M Users Milestone", revenue[19]),
    }

    for month, (label, val) in events.items():
        ax.annotate(
            label,
            xy=(month, val),
            xytext=(month + 1.5, val + 25),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#C00000"),
        )
        ax.plot(month, val, "o", color="#C00000", markersize=8, zorder=5)

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Monthly Revenue ($K)", fontsize=12)
    ax.set_title("Our Growth Story: From Launch to 1M Users",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)

    plt.tight_layout()
    plt.savefig("sample-outputs/chapter-11/storytelling_viz.png", dpi=150, bbox_inches="tight")
    print("Figure saved: sample-outputs/chapter-11/storytelling_viz.png")


if __name__ == "__main__":
    business_dashboard()
    storytelling_visualization()
    plt.show()
