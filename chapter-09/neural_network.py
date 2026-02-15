"""
Chapter 9: Deep Learning and Neural Networks
Section 9.2 — Crafting Neural Networks from the Ground Up

Demonstrates building and training a neural network:
- Constructing a feedforward neural network with Keras
- Binary classification on synthetic data
- Training with callbacks (early stopping)
- Visualizing training history
- Evaluating model performance
"""

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def create_classification_data():
    """Generate synthetic binary classification data."""
    np.random.seed(42)
    n = 1000

    # Two classes with 10 features
    X_class0 = np.random.randn(n // 2, 10) + np.array([1, 0, -1, 0.5, 0, 0, 1, -0.5, 0, 0.5])
    X_class1 = np.random.randn(n // 2, 10) + np.array([-1, 0.5, 1, -0.5, 0.5, 0, -1, 0.5, 0, -0.5])

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # Shuffle
    idx = np.random.permutation(n)
    return X[idx], y[idx]


def build_model(input_dim):
    """Build a feedforward neural network for binary classification."""
    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,), name="hidden_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(32, activation="relu", name="hidden_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(16, activation="relu", name="hidden_3"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_and_evaluate():
    """Full training pipeline with evaluation and visualization."""

    print("=" * 60)
    print("NEURAL NETWORK — Binary Classification")
    print("=" * 60)

    # Prepare data
    X, y = create_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class balance: {(y == 0).sum()} negative, {(y == 1).sum()} positive")

    # Build model
    model = build_model(input_dim=X_train.shape[1])

    print("\n--- Model Architecture ---")
    model.summary()

    # Train with early stopping
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    print("\n--- Training ---")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0  # Suppress epoch-by-epoch output
    )

    actual_epochs = len(history.history["loss"])
    print(f"  Training completed in {actual_epochs} epochs (early stopping)")
    print(f"  Final train loss:      {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # Classification report
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    axes[0].plot(history.history["loss"], label="Train Loss", color="#2B579A")
    axes[0].plot(history.history["val_loss"], label="Validation Loss", color="#C00000")
    axes[0].set_title("Loss Over Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="#2B579A")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#C00000")
    axes[1].set_title("Accuracy Over Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Neural Network Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("nn_training_history.png", dpi=150, bbox_inches="tight")
    print("Figure saved: nn_training_history.png")


if __name__ == "__main__":
    train_and_evaluate()
