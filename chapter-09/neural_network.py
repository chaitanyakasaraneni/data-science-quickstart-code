"""
Chapter 9: Deep Learning and Neural Networks
Section 9.2 — Crafting Neural Networks from the Ground Up

A from-scratch feedforward neural network implemented with NumPy only.
This script intentionally avoids deep-learning frameworks (TensorFlow, PyTorch,
Keras) so that every step described in §9.2 is visible in the code:

  - architecture: the user supplies layer sizes and activation functions
  - weight initialization: He init for ReLU layers, Xavier for sigmoid
  - forward pass: each layer computes z = W a + b, then a = activation(z)
  - loss: binary cross-entropy
  - backward pass: gradients computed by hand via the chain rule
  - parameter update: vanilla mini-batch gradient descent

The script trains on a synthetic 10-feature binary classification problem,
evaluates on a held-out test set, and saves a loss/accuracy curve to
sample-outputs/chapter-09/nn_training_history.png so that the figure
referenced in the manuscript stays in sync with this code.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# --------------------------------------------------------------------------- #
# Activation functions and their derivatives
# --------------------------------------------------------------------------- #

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, None),  # gradient handled jointly with BCE loss
}


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #

class NumpyFeedforwardNN:
    """A plain feedforward neural network for binary classification.

    Architecture is a list of layer sizes including input and output, e.g.
    [10, 16, 8, 1]. Activations is a list one shorter, applied to layers 1..L.
    The output layer must use 'sigmoid' (paired with binary cross-entropy).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        assert len(activations) == len(layer_sizes) - 1, (
            "Need one activation per non-input layer"
        )
        assert activations[-1] == "sigmoid", (
            "Output layer must be sigmoid (paired with binary cross-entropy)"
        )

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = learning_rate

        # Weight initialization: He init for ReLU layers, Xavier for sigmoid.
        # W[l] has shape (n_l, n_{l-1}); b[l] has shape (n_l, 1).
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for l, (n_in, n_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            if activations[l] == "relu":
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)
            self.W.append(rng.standard_normal((n_out, n_in)) * scale)
            self.b.append(np.zeros((n_out, 1)))

    # ------------------------------------------------------------------- #
    # Forward pass: cache (z, a) per layer for use in backprop
    # ------------------------------------------------------------------- #
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list, list]:
        """Run X through the network. Returns (output, list of z, list of a).

        X has shape (n_features, n_samples). Activations are stored with the
        input as a[0] = X so that backprop can reach layer 1.
        """
        a = [X]
        z_cache: list[np.ndarray] = []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ a[-1] + b
            z_cache.append(z)
            act_fn, _ = ACTIVATIONS[self.activations[l]]
            a.append(act_fn(z))
        return a[-1], z_cache, a

    # ------------------------------------------------------------------- #
    # Loss: binary cross-entropy
    # ------------------------------------------------------------------- #
    @staticmethod
    def bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(
            -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        )

    # ------------------------------------------------------------------- #
    # Backward pass: chain rule, layer by layer
    # ------------------------------------------------------------------- #
    def backward(
        self,
        z_cache: list[np.ndarray],
        a_cache: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute gradients dW, db for each layer via backpropagation."""
        m = y_true.shape[1]  # batch size
        L = len(self.W)
        dW: list[np.ndarray | None] = [None] * L
        db: list[np.ndarray | None] = [None] * L

        # Output layer: sigmoid + BCE has the clean gradient dz = a - y
        a_out = a_cache[-1]
        dz = a_out - y_true

        for l in reversed(range(L)):
            a_prev = a_cache[l]  # activation feeding into layer l+1
            dW[l] = (dz @ a_prev.T) / m
            db[l] = np.sum(dz, axis=1, keepdims=True) / m
            if l > 0:
                # Propagate dz back through the previous layer's activation
                da_prev = self.W[l].T @ dz
                _, grad_fn = ACTIVATIONS[self.activations[l - 1]]
                dz = da_prev * grad_fn(z_cache[l - 1])

        return dW, db  # type: ignore[return-value]

    def update(
        self,
        dW: list[np.ndarray],
        db: list[np.ndarray],
    ) -> None:
        for l in range(len(self.W)):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]

    # ------------------------------------------------------------------- #
    # Training loop
    # ------------------------------------------------------------------- #
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose_every: int = 10,
    ) -> dict:
        """Train via mini-batch gradient descent. Returns history dict."""
        # Reshape to (features, samples) and (1, samples) for column-vector math
        X_train = X_train.T
        X_val = X_val.T
        y_train = y_train.reshape(1, -1).astype(float)
        y_val = y_val.reshape(1, -1).astype(float)

        n_samples = X_train.shape[1]
        rng = np.random.default_rng(0)
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        for epoch in range(1, epochs + 1):
            # Shuffle training samples each epoch
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = idx[start : start + batch_size]
                xb, yb = X_train[:, batch], y_train[:, batch]
                _, z_cache, a_cache = self.forward(xb)
                dW, db = self.backward(z_cache, a_cache, yb)
                self.update(dW, db)

            # Epoch-level metrics on full train + val
            train_pred, _, _ = self.forward(X_train)
            val_pred, _, _ = self.forward(X_val)
            train_loss = self.bce_loss(train_pred, y_train)
            val_loss = self.bce_loss(val_pred, y_val)
            train_acc = float(np.mean((train_pred > 0.5) == y_train))
            val_acc = float(np.mean((val_pred > 0.5) == y_val))
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                print(
                    f"  epoch {epoch:3d}/{epochs}  "
                    f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"acc={train_acc:.3f}  val_acc={val_acc:.3f}"
                )

        return history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs, _, _ = self.forward(X.T)
        return (probs.flatten() > threshold).astype(int)


# --------------------------------------------------------------------------- #
# Data + driver
# --------------------------------------------------------------------------- #

def create_classification_data(n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    half = n // 2
    centroid_0 = np.array([1, 0, -1, 0.5, 0, 0, 1, -0.5, 0, 0.5])
    centroid_1 = np.array([-1, 0.5, 1, -0.5, 0.5, 0, -1, 0.5, 0, -0.5])
    X = np.vstack(
        [
            rng.standard_normal((half, 10)) + centroid_0,
            rng.standard_normal((half, 10)) + centroid_1,
        ]
    )
    y = np.array([0] * half + [1] * half)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def train_and_evaluate() -> None:
    print("=" * 60)
    print("NEURAL NETWORK (NumPy from scratch) — Binary Classification")
    print("=" * 60)

    X, y = create_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    net = NumpyFeedforwardNN(
        layer_sizes=[X.shape[1], 16, 8, 1],
        activations=["relu", "relu", "sigmoid"],
        learning_rate=0.05,
        seed=42,
    )

    print("\n--- Architecture ---")
    for l, (W, act) in enumerate(zip(net.W, net.activations), start=1):
        print(f"  layer {l}: W shape {W.shape}, activation {act}")
    n_params = sum(W.size for W in net.W) + sum(b.size for b in net.b)
    print(f"  trainable parameters: {n_params}")

    print("\n--- Training ---")
    history = net.fit(
        X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32, verbose_every=10,
    )

    print("\n--- Test Set Evaluation ---")
    y_pred_test = net.predict(X_test)
    test_acc = float(np.mean(y_pred_test == y_test))
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print(classification_report(y_test, y_pred_test, target_names=["Class 0", "Class 1"]))

    # Plot — match the figure path the book references
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["loss"], label="Train Loss", color="#2B579A")
    axes[0].plot(history["val_loss"], label="Validation Loss", color="#C00000")
    axes[0].set_title("Loss Over Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["accuracy"], label="Train Accuracy", color="#2B579A")
    axes[1].plot(history["val_accuracy"], label="Validation Accuracy", color="#C00000")
    axes[1].set_title("Accuracy Over Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Neural Network Training History (NumPy from scratch)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parents[1] / "sample-outputs" / "chapter-09"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nn_training_history.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    train_and_evaluate()
