"""
Chapter 3: Exploratory Data Analysis (EDA)
Section 3.2 — Techniques for Visualizing Data

Demonstrates key EDA techniques:
- Summary statistics
- Distribution analysis (histograms)
- Relationship exploration (scatter plots)
- Correlation heatmaps
- Box plots for outlier detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_housing_data():
    """Generate a synthetic housing dataset for EDA demonstration."""
    np.random.seed(42)
    n = 500

    sqft = np.random.normal(1800, 500, n).clip(600, 5000).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.4, 0.25, 0.1])
    bathrooms = np.clip(bedrooms - np.random.choice([0, 1], n, p=[0.6, 0.4]), 1, 4)
    age = np.random.randint(0, 50, n)

    # Price with realistic relationships
    price = (
        50000
        + 150 * sqft
        + 20000 * bedrooms
        + 15000 * bathrooms
        - 1000 * age
        + np.random.normal(0, 30000, n)
    ).clip(50000).astype(int)

    # Add a few outliers
    price[np.random.choice(n, 5)] = np.random.randint(800000, 1200000, 5)

    return pd.DataFrame({
        "price": price,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
    })


def perform_eda(df):
    """Run a complete EDA workflow with visualizations."""

    # --- Summary Statistics ---
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe().round(2))
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # --- Figure 1: Distribution and Relationships ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Exploratory Data Analysis — Housing Dataset", fontsize=14, fontweight="bold")

    # 1. Histogram — Price Distribution
    axes[0, 0].hist(df["price"], bins=30, edgecolor="black", color="#2B579A", alpha=0.8)
    axes[0, 0].set_title("Distribution of House Prices")
    axes[0, 0].set_xlabel("Price ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(df["price"].median(), color="red", linestyle="--", label=f"Median: ${df['price'].median():,.0f}")
    axes[0, 0].legend()

    # 2. Scatter Plot — Price vs Square Footage
    axes[0, 1].scatter(df["sqft"], df["price"], alpha=0.4, s=15, color="#2B579A")
    axes[0, 1].set_title("Price vs. Square Footage")
    axes[0, 1].set_xlabel("Square Feet")
    axes[0, 1].set_ylabel("Price ($)")
    # Add trend line
    z = np.polyfit(df["sqft"], df["price"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["sqft"].min(), df["sqft"].max(), 100)
    axes[0, 1].plot(x_line, p(x_line), "r--", linewidth=2, label="Trend Line")
    axes[0, 1].legend()

    # 3. Box Plot — Price by Bedrooms
    bedroom_groups = [df[df["bedrooms"] == b]["price"].values for b in sorted(df["bedrooms"].unique())]
    bp = axes[1, 0].boxplot(bedroom_groups, labels=sorted(df["bedrooms"].unique()), patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2B579A")
        patch.set_alpha(0.6)
    axes[1, 0].set_title("Price Distribution by Bedrooms")
    axes[1, 0].set_xlabel("Number of Bedrooms")
    axes[1, 0].set_ylabel("Price ($)")

    # 4. Correlation Heatmap
    corr = df.corr()
    im = axes[1, 1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(corr.columns)))
    axes[1, 1].set_yticks(range(len(corr.columns)))
    axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    axes[1, 1].set_yticklabels(corr.columns, fontsize=9)
    axes[1, 1].set_title("Correlation Heatmap")
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}",
                           ha="center", va="center", fontsize=8,
                           color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("eda_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved as 'eda_output.png'")

    # --- Key Findings ---
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"  Strongest positive correlation: sqft → price ({corr['price']['sqft']:.3f})")
    print(f"  Median house price: ${df['price'].median():,.0f}")
    print(f"  Price range: ${df['price'].min():,.0f} — ${df['price'].max():,.0f}")
    print(f"  Potential outliers: {(df['price'] > df['price'].quantile(0.99)).sum()} houses above 99th percentile")


if __name__ == "__main__":
    df = create_housing_data()
    perform_eda(df)
