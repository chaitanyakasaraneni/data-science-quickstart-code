"""
Chapter 2: Data Collection and Management
Section 2.2 — Data Cleaning: Ensuring Data Quality

Demonstrates fundamental data cleaning techniques using pandas:
- Detecting and handling missing values
- Removing duplicate records
- Standardizing text columns
- Basic data validation
"""

import pandas as pd
import numpy as np


def create_sample_data():
    """Generate a synthetic sales dataset with common data quality issues."""
    print("=" * 60)
    print("CREATING SAMPLE DATA WITH QUALITY ISSUES")
    np.random.seed(42)
    n = 200

    data = {
        "order_id": range(1, n + 1),
        "customer_name": np.random.choice(
            ["Alice Johnson", "bob smith", "  Charlie Brown ", "Diana Prince",
             "Eve Wilson", "Frank Miller", None], n
        ),
        "region": np.random.choice(
            ["north", "SOUTH", " East", "west ", "North", "south"], n
        ),
        "product": np.random.choice(
            ["Widget A", "Widget B", "Widget C", "Widget D"], n
        ),
        "quantity": np.random.randint(1, 50, n).astype(float),
        "revenue": np.random.uniform(10, 5000, n).round(2),
    }

    df = pd.DataFrame(data)

    # Introduce missing values
    missing_indices = np.random.choice(n, size=15, replace=False)
    df.loc[missing_indices[:8], "revenue"] = np.nan
    df.loc[missing_indices[8:], "quantity"] = np.nan

    # Introduce duplicates
    duplicates = df.sample(10, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    print("=" * 60)
    print("Sample data created with the following issues:")
    print(f"  - Missing values in 'revenue': {df['revenue'].isnull().sum()}")
    print(f"  - Missing values in 'quantity': {df['quantity'].isnull().sum()}")
    print(f"  - Duplicate rows: {df.duplicated().sum()}")
    print("=" * 60)

    return df


def clean_data(df):
    """Apply a standard data cleaning pipeline."""

    print("=" * 60)
    print("STEP 1: Initial Assessment")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Step 2: Handle missing values
    print("\n" + "=" * 60)
    print("STEP 2: Handle Missing Values")
    print("=" * 60)

    # Fill numeric columns with the median
    for col in ["revenue", "quantity"]:
        median_val = df[col].median()
        missing_count = df[col].isnull().sum()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {missing_count} missing '{col}' values with median ({median_val:.2f})")

    # Fill categorical columns with 'Unknown'
    missing_names = df["customer_name"].isnull().sum()
    df["customer_name"] = df["customer_name"].fillna("Unknown")
    print(f"  Filled {missing_names} missing 'customer_name' values with 'Unknown'")

    # Step 3: Remove duplicates
    print("\n" + "=" * 60)
    print("STEP 3: Remove Duplicates")
    print("=" * 60)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"  Removed {before - after} duplicate rows ({before} → {after})")

    # Step 4: Standardize text columns
    print("\n" + "=" * 60)
    print("STEP 4: Standardize Text Columns")
    print("=" * 60)

    df["customer_name"] = df["customer_name"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()

    print(f"  Unique regions after standardization: {sorted(df['region'].unique())}")
    print(f"  Sample customer names: {df['customer_name'].head(5).tolist()}")

    # Step 5: Validate data
    print("\n" + "=" * 60)
    print("STEP 5: Validate Cleaned Data")
    print("=" * 60)

    # Ensure no negative values
    invalid_qty = (df["quantity"] <= 0).sum()
    invalid_rev = (df["revenue"] <= 0).sum()
    print(f"  Invalid quantities (<=0): {invalid_qty}")
    print(f"  Invalid revenue (<=0): {invalid_rev}")

    if invalid_qty > 0:
        df = df[df["quantity"] > 0]
    if invalid_rev > 0:
        df = df[df["revenue"] > 0]

    print(f"\n  Final cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")

    return df


if __name__ == "__main__":
    # Generate sample data with quality issues
    raw_df = create_sample_data()

    # Apply cleaning pipeline
    clean_df = clean_data(raw_df)

    # Preview cleaned data
    print("\n" + "=" * 60)
    print("CLEANED DATA PREVIEW")
    print("=" * 60)
    print(clean_df.head(10).to_string(index=False))
