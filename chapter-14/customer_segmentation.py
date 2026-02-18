"""
Chapter 14: Data Science for Business Analytics
Section 14.3 — Customer Behavior and Sentiment Analysis

Demonstrates customer segmentation using RFM analysis and K-Means:
- RFM (Recency, Frequency, Monetary) feature engineering
- K-Means clustering for customer segments
- Segment profiling and business interpretation
- Visualization of customer segments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def create_customer_data():
    """Generate synthetic e-commerce customer transaction data."""
    np.random.seed(42)
    n_customers = 500

    # Customer IDs
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

    # Simulate different customer behaviors
    segments = np.random.choice(
        ["loyal", "at_risk", "new", "big_spender", "occasional"],
        n_customers,
        p=[0.2, 0.15, 0.25, 0.1, 0.3]
    )

    data = []
    for cid, seg in zip(customer_ids, segments):
        if seg == "loyal":
            recency = np.random.randint(1, 30)
            frequency = np.random.randint(15, 50)
            monetary = np.random.uniform(500, 2000)
        elif seg == "at_risk":
            recency = np.random.randint(60, 180)
            frequency = np.random.randint(5, 20)
            monetary = np.random.uniform(200, 800)
        elif seg == "new":
            recency = np.random.randint(1, 30)
            frequency = np.random.randint(1, 5)
            monetary = np.random.uniform(50, 300)
        elif seg == "big_spender":
            recency = np.random.randint(5, 60)
            frequency = np.random.randint(8, 25)
            monetary = np.random.uniform(2000, 8000)
        else:  # occasional
            recency = np.random.randint(30, 120)
            frequency = np.random.randint(1, 8)
            monetary = np.random.uniform(50, 500)

        data.append({
            "customer_id": cid,
            "days_since_last_purchase": recency,
            "total_purchases": frequency,
            "total_spend": round(monetary, 2),
        })

    return pd.DataFrame(data)


def rfm_analysis(df):
    """Perform RFM analysis and K-Means segmentation."""

    print("=" * 60)
    print("CUSTOMER SEGMENTATION — RFM Analysis with K-Means")
    print("=" * 60)

    print(f"\nDataset: {len(df)} customers")
    print(f"\nRFM Summary Statistics:")
    print(df[["days_since_last_purchase", "total_purchases", "total_spend"]].describe().round(2))

    # Prepare RFM features
    rfm = df[["days_since_last_purchase", "total_purchases", "total_spend"]].copy()
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Find optimal K using elbow method
    print("\n--- Elbow Method ---")
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)
        print(f"  K={k}: Inertia={km.inertia_:.1f}")

    # Use K=4 (typically good for RFM)
    optimal_k = 4
    print(f"\n  Selected K={optimal_k} based on elbow analysis")

    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["segment"] = kmeans.fit_predict(rfm_scaled)

    # Profile each segment
    print("\n\n--- Segment Profiles ---")
    segment_profiles = df.groupby("segment").agg({
        "customer_id": "count",
        "days_since_last_purchase": "mean",
        "total_purchases": "mean",
        "total_spend": "mean",
    }).round(1)
    segment_profiles.columns = ["Count", "Avg Recency (days)", "Avg Frequency", "Avg Spend ($)"]

    # Label segments based on characteristics
    labels = {}
    for seg in range(optimal_k):
        profile = segment_profiles.loc[seg]
        recency = profile["Avg Recency (days)"]
        frequency = profile["Avg Frequency"]
        spend = profile["Avg Spend ($)"]

        if spend > 2000:
            labels[seg] = "💎 VIP / High Value"
        elif recency < 25 and frequency > 15:
            labels[seg] = "⭐ Loyal Customers"
        elif recency < 25 and frequency < 5:
            labels[seg] = "🆕 New Customers"
        elif recency > 60:
            labels[seg] = "⚠️ At Risk / Churning"
        else:
            labels[seg] = "📊 Regular"

    segment_profiles["Label"] = [labels[i] for i in range(optimal_k)]

    for seg in range(optimal_k):
        p = segment_profiles.loc[seg]
        print(f"\n  Segment {seg}: {p['Label']}")
        print(f"    Customers:      {int(p['Count'])}")
        print(f"    Avg Recency:    {p['Avg Recency (days)']:.0f} days")
        print(f"    Avg Frequency:  {p['Avg Frequency']:.1f} purchases")
        print(f"    Avg Spend:      ${p['Avg Spend ($)']:,.2f}")

    # Business recommendations
    print("\n\n--- Business Recommendations ---")
    for seg in range(optimal_k):
        label = labels[seg]
        if "VIP" in label:
            print(f"\n  {label}:")
            print(f"    → Offer exclusive rewards and early access to new products")
            print(f"    → Assign dedicated account managers")
        elif "Loyal" in label:
            print(f"\n  {label}:")
            print(f"    → Launch loyalty program with tiered benefits")
            print(f"    → Cross-sell and upsell premium products")
        elif "New" in label:
            print(f"\n  {label}:")
            print(f"    → Send welcome series with onboarding guides")
            print(f"    → Offer first-purchase discounts to drive repeat visits")
        elif "At Risk" in label:
            print(f"\n  {label}:")
            print(f"    → Trigger win-back email campaigns with incentives")
            print(f"    → Conduct churn analysis to identify pain points")
        else:
            print(f"\n  {label}:")
            print(f"    → Maintain engagement with periodic promotions")
            print(f"    → Encourage higher purchase frequency")

    return df, rfm_scaled, labels


def visualize_segments(df, rfm_scaled, labels):
    """Create visualizations of customer segments."""
    print("\n\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ["#2B579A", "#C00000", "#2E8B57", "#FF8C00"]
    segment_names = [labels[i].split(" ", 1)[1] for i in range(len(labels))]

    # 1. Recency vs Frequency
    for seg in sorted(df["segment"].unique()):
        mask = df["segment"] == seg
        axes[0].scatter(
            df.loc[mask, "days_since_last_purchase"],
            df.loc[mask, "total_purchases"],
            c=colors[seg], s=20, alpha=0.6, label=segment_names[seg]
        )
    axes[0].set_xlabel("Recency (days since last purchase)")
    axes[0].set_ylabel("Frequency (total purchases)")
    axes[0].set_title("Recency vs Frequency", fontweight="bold")
    axes[0].legend(fontsize=7)

    # 2. Frequency vs Monetary
    for seg in sorted(df["segment"].unique()):
        mask = df["segment"] == seg
        axes[1].scatter(
            df.loc[mask, "total_purchases"],
            df.loc[mask, "total_spend"],
            c=colors[seg], s=20, alpha=0.6, label=segment_names[seg]
        )
    axes[1].set_xlabel("Frequency (total purchases)")
    axes[1].set_ylabel("Monetary (total spend $)")
    axes[1].set_title("Frequency vs Monetary", fontweight="bold")
    axes[1].legend(fontsize=7)

    # 3. Segment size bar chart
    seg_counts = df["segment"].value_counts().sort_index()
    bars = axes[2].bar(
        [segment_names[i] for i in seg_counts.index],
        seg_counts.values,
        color=[colors[i] for i in seg_counts.index],
        edgecolor="black", linewidth=0.5
    )
    axes[2].set_ylabel("Number of Customers")
    axes[2].set_title("Segment Distribution", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=20)
    for bar, count in zip(bars, seg_counts.values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Customer Segmentation Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample-outputs/chapter-14/customer_segments.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: sample-outputs/chapter-14/customer_segments.png")


if __name__ == "__main__":
    df = create_customer_data()
    df, rfm_scaled, labels = rfm_analysis(df)
    visualize_segments(df, rfm_scaled, labels)
