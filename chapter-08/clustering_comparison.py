"""
Chapter 8: Unsupervised Learning Techniques
Section 8.2 — Clustering Techniques: K-Means, DBSCAN

Demonstrates and visually compares K-Means and DBSCAN:
- K-Means on well-separated spherical clusters
- DBSCAN on non-linear/irregular cluster shapes
- Side-by-side comparison on the same dataset
- Elbow method for choosing K
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler


def compare_on_moons():
    """Compare K-Means and DBSCAN on crescent-shaped (non-linear) data."""
    print("=" * 60)
    print("COMPARISON 1: Crescent-Shaped Data (make_moons)")
    print("=" * 60)
    print("This dataset has two interleaving crescent shapes —")
    print("ideal for demonstrating DBSCAN's advantage.\n")

    X, y_true = make_moons(n_samples=400, noise=0.08, random_state=42)

    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    db_labels = dbscan.fit_predict(X)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="Set1", s=15, alpha=0.7)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")

    axes[1].scatter(X[:, 0], X[:, 1], c=km_labels, cmap="Set1", s=15, alpha=0.7)
    axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    c="black", marker="X", s=150, edgecolors="white", linewidths=2)
    axes[1].set_title("K-Means (K=2)", fontsize=12, fontweight="bold")

    noise_mask = db_labels == -1
    axes[2].scatter(X[~noise_mask, 0], X[~noise_mask, 1],
                    c=db_labels[~noise_mask], cmap="Set1", s=15, alpha=0.7)
    axes[2].scatter(X[noise_mask, 0], X[noise_mask, 1],
                    c="gray", marker="x", s=30, alpha=0.5, label="Noise")
    axes[2].set_title("DBSCAN (ε=0.2, MinPts=5)", fontsize=12, fontweight="bold")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.suptitle("K-Means vs DBSCAN on Non-Linear Data", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("sample-outputs/chapter-08/clustering_moons.png", dpi=150, bbox_inches="tight")
    print("Figure saved: sample-outputs/chapter-08/clustering_moons.png")

    # Print analysis
    n_noise = (db_labels == -1).sum()
    print(f"\n  K-Means clusters: {len(set(km_labels))}")
    print(f"  DBSCAN clusters:  {len(set(db_labels) - {-1})}, noise points: {n_noise}")
    print(f"\n  Key insight: K-Means splits the data with a linear boundary,")
    print(f"  misassigning points where the crescents interleave. DBSCAN")
    print(f"  correctly identifies the two non-linear clusters.")


def compare_on_blobs():
    """Compare on well-separated spherical clusters (K-Means' strength)."""
    print("\n\n" + "=" * 60)
    print("COMPARISON 2: Well-Separated Spherical Clusters (make_blobs)")
    print("=" * 60)
    print("This dataset has compact, spherical clusters —")
    print("ideal for K-Means.\n")

    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
    X = StandardScaler().fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(eps=0.4, min_samples=5)
    db_labels = dbscan.fit_predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="Set2", s=15, alpha=0.7)
    axes[0].set_title("Ground Truth (4 clusters)", fontsize=12, fontweight="bold")

    axes[1].scatter(X[:, 0], X[:, 1], c=km_labels, cmap="Set2", s=15, alpha=0.7)
    axes[1].set_title("K-Means (K=4)", fontsize=12, fontweight="bold")

    noise_mask = db_labels == -1
    axes[2].scatter(X[~noise_mask, 0], X[~noise_mask, 1],
                    c=db_labels[~noise_mask], cmap="Set2", s=15, alpha=0.7)
    if noise_mask.any():
        axes[2].scatter(X[noise_mask, 0], X[noise_mask, 1],
                        c="gray", marker="x", s=30, alpha=0.5, label="Noise")
        axes[2].legend()
    n_clusters_db = len(set(db_labels) - {-1})
    axes[2].set_title(f"DBSCAN ({n_clusters_db} clusters found)", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.suptitle("K-Means vs DBSCAN on Spherical Clusters", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("sample-outputs/chapter-08/clustering_blobs.png", dpi=150, bbox_inches="tight")
    print("Figure saved: sample-outputs/chapter-08/clustering_blobs.png")

    print(f"\n  K-Means clusters: {len(set(km_labels))}")
    print(f"  DBSCAN clusters:  {n_clusters_db}, noise points: {noise_mask.sum()}")
    print(f"\n  Key insight: Both algorithms perform well on spherical data,")
    print(f"  but K-Means is simpler and more efficient for this case.")


def elbow_method():
    """Demonstrate the elbow method for choosing K in K-Means."""
    print("\n\n" + "=" * 60)
    print("BONUS: Elbow Method for Choosing K")
    print("=" * 60)

    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
    X = StandardScaler().fit_transform(X)

    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  K={k:2d}  Inertia={km.inertia_:8.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, "o-", color="#2B579A", linewidth=2, markersize=8)
    plt.axvline(x=4, color="red", linestyle="--", alpha=0.7, label="Optimal K=4")
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sample-outputs/chapter-08/elbow_method.png", dpi=150)
    print("\nFigure saved: sample-outputs/chapter-08/elbow_method.png")
    print("\n  The 'elbow' at K=4 indicates the optimal number of clusters,")
    print("  where adding more clusters yields diminishing returns.")


if __name__ == "__main__":
    compare_on_moons()
    compare_on_blobs()
    elbow_method()
    plt.show()
