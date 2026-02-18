"""
Chapter 4: Data Wrangling and Transformation
Section 4.2 — Techniques for Data Transformation

Demonstrates common data transformation techniques:
- Column renaming and standardization
- Date parsing and feature extraction
- Min-max normalization
- One-hot encoding
- Binning continuous variables
- Merging and reshaping data
"""

import pandas as pd
import numpy as np


def create_raw_data():
    """Generate synthetic raw e-commerce data with common issues."""
    np.random.seed(42)
    n = 100

    data = {
        "Order ID": range(1001, 1001 + n),
        "Customer Name": np.random.choice(
            ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince",
             "Eve Wilson", "Frank Miller", "Grace Lee", "Henry Davis"], n
        ),
        "Order Date": pd.date_range("2023-01-05", periods=n, freq="3D").strftime("%m/%d/%Y"),
        "Product Category": np.random.choice(
            ["electronics", "CLOTHING", "Home & Garden", "books", "Sports"], n
        ),
        "Amount": np.random.uniform(5, 500, n).round(2),
        "Quantity": np.random.randint(1, 10, n),
        "Customer Age": np.random.randint(18, 70, n),
        "Shipping City": np.random.choice(
            ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "Philadelphia", "San Antonio", "San Diego"], n
        ),
    }

    return pd.DataFrame(data)


def transform_data(df):
    """Apply a comprehensive transformation pipeline."""

    print("=" * 60)
    print("ORIGINAL DATA")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))

    # --- Step 1: Rename columns to snake_case ---
    print("\n" + "=" * 60)
    print("STEP 1: Standardize Column Names")
    print("=" * 60)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")
    print(f"  New columns: {list(df.columns)}")

    # --- Step 2: Parse dates and extract features ---
    print("\n" + "=" * 60)
    print("STEP 2: Parse Dates and Extract Temporal Features")
    print("=" * 60)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_day_of_week"] = df["order_date"].dt.day_name()
    df["order_quarter"] = df["order_date"].dt.quarter
    print(f"  Extracted: order_year, order_month, order_day_of_week, order_quarter")
    print(f"  Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")

    # --- Step 3: Standardize categorical text ---
    print("\n" + "=" * 60)
    print("STEP 3: Standardize Categorical Text")
    print("=" * 60)
    df["product_category"] = df["product_category"].str.strip().str.title()
    print(f"  Categories: {sorted(df['product_category'].unique())}")

    # --- Step 4: Create derived features ---
    print("\n" + "=" * 60)
    print("STEP 4: Create Derived Features")
    print("=" * 60)
    df["total_value"] = df["amount"] * df["quantity"]
    df["avg_item_price"] = (df["amount"] / df["quantity"]).round(2)
    print(f"  Added: total_value, avg_item_price")
    print(f"  Total value range: ${df['total_value'].min():.2f} — ${df['total_value'].max():.2f}")

    # --- Step 5: Min-Max Normalization ---
    print("\n" + "=" * 60)
    print("STEP 5: Normalize Numeric Features (Min-Max Scaling)")
    print("=" * 60)
    for col in ["amount", "quantity", "customer_age"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_normalized"] = ((df[col] - col_min) / (col_max - col_min)).round(4)
        print(f"  {col}: [{col_min} — {col_max}] → [0.0 — 1.0]")

    # --- Step 6: Bin continuous variable ---
    print("\n" + "=" * 60)
    print("STEP 6: Bin Continuous Variables")
    print("=" * 60)
    df["age_group"] = pd.cut(
        df["customer_age"],
        bins=[0, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "51+"]
    )
    print(f"  Age group distribution:")
    print(df["age_group"].value_counts().sort_index().to_string())

    df["spending_tier"] = pd.cut(
        df["total_value"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Low", "Medium", "High", "Premium"]
    )
    print(f"\n  Spending tier distribution:")
    print(df["spending_tier"].value_counts().sort_index().to_string())

    # --- Step 7: One-Hot Encoding ---
    print("\n" + "=" * 60)
    print("STEP 7: One-Hot Encode Categorical Features")
    print("=" * 60)
    df_encoded = pd.get_dummies(df, columns=["product_category"], prefix="cat", dtype=int)
    new_cols = [c for c in df_encoded.columns if c.startswith("cat_")]
    print(f"  Created dummy columns: {new_cols}")

    return df_encoded


def summarize_transformations(df_original, df_transformed):
    """Print a summary of all transformations applied."""
    print("\n" + "=" * 60)
    print("TRANSFORMATION SUMMARY")
    print("=" * 60)
    print(f"  Original: {df_original.shape[0]} rows × {df_original.shape[1]} columns")
    print(f"  Transformed: {df_transformed.shape[0]} rows × {df_transformed.shape[1]} columns")
    print(f"  New columns added: {df_transformed.shape[1] - df_original.shape[1]}")
    print(f"\n  Final columns:")
    for i, col in enumerate(df_transformed.columns, 1):
        print(f"    {i:2d}. {col} ({df_transformed[col].dtype})")


if __name__ == "__main__":
    raw_df = create_raw_data()
    transformed_df = transform_data(raw_df)
    summarize_transformations(raw_df, transformed_df)

    # Save transformed data
    transformed_df.to_csv("sample-outputs/chapter-04/transformed_data.csv", index=False)
    print("\nTransformed data saved to 'sample-outputs/chapter-04/transformed_data.csv'")
