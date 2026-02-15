"""
Chapter 18: The Role of AI in Data Science
Section 18.2 — Automation of Data Science Tasks with AI

Demonstrates automated machine learning concepts:
- Automated hyperparameter tuning with GridSearchCV
- Automated model selection across multiple algorithms
- Pipeline automation (preprocessing + model)
- Comparison of manual vs automated approaches
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time


def create_dataset():
    """Generate a synthetic classification dataset."""
    np.random.seed(42)
    n = 800

    # 8 features with varying relevance
    X = np.random.randn(n, 8)

    # Target depends on a subset of features
    logits = (
        1.5 * X[:, 0] - 0.8 * X[:, 1] + 1.2 * X[:, 2]
        - 0.5 * X[:, 3] + 0.3 * X[:, 0] * X[:, 2]
        + np.random.normal(0, 0.5, n)
    )
    y = (logits > 0).astype(int)

    feature_names = [f"feature_{i+1}" for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def manual_approach(X_train, X_test, y_train, y_test):
    """Traditional manual model training with default hyperparameters."""
    print("=" * 60)
    print("APPROACH 1: Manual (Default Hyperparameters)")
    print("=" * 60)

    start = time.time()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start

    print(f"\n  Model: Random Forest (default params)")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return accuracy


def automated_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """Automated hyperparameter search using GridSearchCV."""
    print("\n\n" + "=" * 60)
    print("APPROACH 2: Automated Hyperparameter Tuning (GridSearchCV)")
    print("=" * 60)

    start = time.time()

    # Create pipeline: scaling + model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameter grid
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"\n  Hyperparameter grid: {total_combinations} combinations × 5-fold CV")
    print(f"  = {total_combinations * 5} total model fits\n")

    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring="accuracy",
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    elapsed = time.time() - start

    print(f"  Best parameters:")
    for param, value in grid_search.best_params_.items():
        clean_param = param.replace("model__", "")
        print(f"    {clean_param}: {value}")

    print(f"\n  Best CV accuracy: {grid_search.best_score_:.4f}")

    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy:    {test_accuracy:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return test_accuracy


def automated_model_selection(X_train, X_test, y_train, y_test):
    """Automated model selection across multiple algorithms."""
    print("\n\n" + "=" * 60)
    print("APPROACH 3: Automated Model Selection")
    print("=" * 60)
    print("  Comparing multiple algorithms with cross-validation...\n")

    start = time.time()

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=42))
        ]),
    }

    results = []
    print(f"  {'Model':<25} {'CV Mean':>8} {'CV Std':>8} {'Test Acc':>9}")
    print("  " + "-" * 52)

    for name, pipeline in models.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))

        results.append({
            "model": name,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_acc": test_acc
        })

        print(f"  {name:<25} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f} {test_acc:>9.4f}")

    elapsed = time.time() - start

    # Find best model
    best = max(results, key=lambda x: x["cv_mean"])
    print(f"\n  Best model: {best['model']} (CV: {best['cv_mean']:.4f})")
    print(f"  Total time for all models: {elapsed:.2f}s")

    # Tune the best model
    print(f"\n  Now tuning {best['model']} with GridSearchCV...")

    if "Gradient" in best["model"]:
        best_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=42))
        ])
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        }
    else:
        best_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42))
        ])
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
        }

    tuned = GridSearchCV(best_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    tuned.fit(X_train, y_train)
    final_acc = accuracy_score(y_test, tuned.predict(X_test))

    print(f"  Tuned test accuracy: {final_acc:.4f}")

    return final_acc


if __name__ == "__main__":
    # Create dataset
    df = create_dataset()
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset: {len(df)} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class balance: {(y == 0).sum()} / {(y == 1).sum()}\n")

    # Run all three approaches
    acc_manual = manual_approach(X_train, X_test, y_train, y_test)
    acc_tuned = automated_hyperparameter_tuning(X_train, X_test, y_train, y_test)
    acc_selected = automated_model_selection(X_train, X_test, y_train, y_test)

    # Final comparison
    print("\n\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\n  {'Approach':<40} {'Test Accuracy':>13}")
    print("  " + "-" * 55)
    print(f"  {'Manual (RF, default params)':<40} {acc_manual:>13.4f}")
    print(f"  {'Automated Hyperparameter Tuning':<40} {acc_tuned:>13.4f}")
    print(f"  {'Automated Model Selection + Tuning':<40} {acc_selected:>13.4f}")
    print(f"\n  Automation improved accuracy by {(acc_selected - acc_manual)*100:.2f} percentage points")
    print(f"  while systematically exploring a much larger search space.")
