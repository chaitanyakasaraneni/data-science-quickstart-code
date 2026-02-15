"""
Chapter 7: Supervised Learning Techniques
Section 7.2 — Key Algorithms: Regression and Classification

Demonstrates supervised learning with practical examples:
- Linear Regression for price prediction
- Logistic Regression for binary classification
- Model evaluation with proper train/test split
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler


def create_housing_data():
    """Generate synthetic housing data for regression."""
    np.random.seed(42)
    n = 500

    sqft = np.random.normal(1800, 500, n).clip(600, 5000).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.4, 0.25, 0.1])
    bathrooms = np.clip(bedrooms - np.random.choice([0, 1], n, p=[0.6, 0.4]), 1, 4)
    age = np.random.randint(0, 50, n)
    garage = np.random.choice([0, 1], n, p=[0.3, 0.7])

    price = (
        50000 + 150 * sqft + 20000 * bedrooms
        + 15000 * bathrooms - 1000 * age + 25000 * garage
        + np.random.normal(0, 25000, n)
    ).clip(50000).astype(int)

    return pd.DataFrame({
        "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age, "has_garage": garage, "price": price
    })


def linear_regression_demo():
    """Demonstrate linear regression for house price prediction."""
    print("=" * 60)
    print("LINEAR REGRESSION — House Price Prediction")
    print("=" * 60)

    df = create_housing_data()
    print(f"\nDataset: {df.shape[0]} houses, {df.shape[1]} features")
    print(df.describe().round(1))

    # Prepare features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"\n--- Model Performance ---")
    print(f"  R² Score:  {r2:.4f} ({r2*100:.1f}% of variance explained)")
    print(f"  RMSE:      ${rmse:,.0f}")
    print(f"  MAE:       ${mae:,.0f}")

    # Feature importance (coefficients)
    print(f"\n--- Feature Coefficients ---")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_.round(2)
    }).sort_values("Coefficient", key=abs, ascending=False)
    for _, row in coef_df.iterrows():
        direction = "+" if row["Coefficient"] > 0 else ""
        print(f"  {row['Feature']:<15} {direction}{row['Coefficient']:>10,.2f}")
    print(f"  {'Intercept':<15} {model.intercept_:>10,.2f}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    print(f"  Each additional sqft adds ~${model.coef_[0]:,.0f} to the price")
    print(f"  Each additional bedroom adds ~${model.coef_[1]:,.0f}")
    print(f"  Having a garage adds ~${model.coef_[4]:,.0f}")
    print(f"  Each year of age reduces price by ~${abs(model.coef_[3]):,.0f}")

    # Sample predictions
    print(f"\n--- Sample Predictions vs Actual ---")
    comparison = pd.DataFrame({
        "Actual": y_test.values[:5],
        "Predicted": y_pred[:5].astype(int),
        "Error": (y_test.values[:5] - y_pred[:5].astype(int))
    })
    print(comparison.to_string(index=False))


def create_classification_data():
    """Generate synthetic customer churn data for classification."""
    np.random.seed(42)
    n = 600

    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 100, n).round(2)
    support_calls = np.random.poisson(2, n)
    contract_type = np.random.choice([0, 1], n, p=[0.4, 0.6])  # 0=monthly, 1=annual

    # Churn probability based on features
    churn_prob = 1 / (1 + np.exp(-(
        -2 + 0.03 * monthly_charges - 0.05 * tenure
        + 0.3 * support_calls - 1.5 * contract_type
        + np.random.normal(0, 0.5, n)
    )))
    churned = (churn_prob > 0.5).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "annual_contract": contract_type,
        "churned": churned
    })


def logistic_regression_demo():
    """Demonstrate logistic regression for customer churn prediction."""
    print("\n\n" + "=" * 60)
    print("LOGISTIC REGRESSION — Customer Churn Prediction")
    print("=" * 60)

    df = create_classification_data()
    print(f"\nDataset: {df.shape[0]} customers")
    print(f"Churn rate: {df['churned'].mean()*100:.1f}%")

    X = df.drop("churned", axis=1)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n--- Model Performance ---")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    print(f"--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted Retained  Predicted Churned")
    print(f"  Actual Retained    {cm[0,0]:>10}    {cm[0,1]:>14}")
    print(f"  Actual Churned     {cm[1,0]:>10}    {cm[1,1]:>14}")

    # Feature importance
    print(f"\n--- Feature Importance (Coefficients) ---")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0].round(4)
    }).sort_values("Coefficient", key=abs, ascending=False)
    for _, row in coef_df.iterrows():
        effect = "↑ churn risk" if row["Coefficient"] > 0 else "↓ churn risk"
        print(f"  {row['Feature']:<20} {row['Coefficient']:>8.4f}  ({effect})")


if __name__ == "__main__":
    linear_regression_demo()
    logistic_regression_demo()
