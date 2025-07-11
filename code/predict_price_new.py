import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def load_data(path: str, input_cols=(2, 3), target_col=4):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X = data[:, input_cols]
    y = data[:, target_col].reshape(-1, 1)
    return X, y

def standardsize(X):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def train_mlp(X, y, hidden_dim=16, lr=0.01, epochs=1000, reg_lambda=0.001):
    n_samples, n_features = X.shape
    W1 = np.random.randn(n_features, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * 0.1
    b2 = np.zeros((1, 1))
    losses = []

    for i in range(epochs):
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        y_pred = Z2

        mse_loss = np.mean((y_pred - y) ** 2) / 2
        reg_loss = reg_lambda * (np.sum(W1**2) + np.sum(W2**2))
        loss = mse_loss + reg_loss
        losses.append(loss)

        dZ2 = (y_pred - y) / n_samples
        dW2 = A1.T @ dZ2 + reg_lambda * W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1 + reg_lambda * W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if i % 100 == 0:
            print(f"Epoch {i} | Loss {loss:.4f}")

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "losses": losses}

def print_image(X_raw, y, y_pred, losses):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(6, 4))
    plt.plot(losses, linewidth=1.8, color='crimson')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练误差")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mlp_loss.png", dpi=300)
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_raw[:, 0], X_raw[:, 1], y.flatten(), c='royalblue', alpha=0.7, s=12)
    ax.set_xlabel("房龄")
    ax.set_ylabel("收入")
    ax.set_zlabel("真实房价")
    ax.set_title("3D散点图：真实房价（原始坐标）")
    plt.tight_layout()
    plt.savefig("mlp_true.png", dpi=300)
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    grid_x, grid_y = np.meshgrid(
        np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 100),
        np.linspace(X_raw[:, 1].min(), X_raw[:, 1].max(), 100)
    )
    grid_z = griddata(X_raw, y_pred.flatten(), (grid_x, grid_y), method='cubic')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.8)
    ax.set_xlabel("房龄")
    ax.set_ylabel("收入")
    ax.set_zlabel("预测房价")
    ax.set_title("3D曲面图：预测房价（原始坐标）")
    plt.tight_layout()
    plt.savefig("mlp_pred.png", dpi=300)
    plt.show()

def run_mlp(csv_path="MLP_data.csv", 
            feature_cols=(2, 3), 
            target_col=4, 
            hidden_dim=16,
            lr=0.005, 
            epochs=2000, 
            reg_lambda=0.005, 
            price_upper_limit=500000):
    
    X_raw, y_raw = load_data(csv_path, feature_cols, target_col)
    mask = y_raw.flatten() < price_upper_limit
    X_raw, y_raw = X_raw[mask], y_raw[mask]

    y_mean, y_std = y_raw.mean(), y_raw.std()
    y = (y_raw - y_mean) / y_std

    X_norm, mu, sigma = standardsize(X_raw)
    model = train_mlp(X_norm, y, hidden_dim=hidden_dim, lr=lr, epochs=epochs, reg_lambda=reg_lambda)
    Z1 = X_norm @ model["W1"] + model["b1"]
    A1 = relu(Z1)
    y_pred = A1 @ model["W2"] + model["b2"]
    y_pred_rescaled = y_pred * y_std + y_mean

    print_image(X_raw, y_raw, y_pred_rescaled, model["losses"])
    return model, mu, sigma, y_mean, y_std

if __name__ == "__main__":
    run_mlp()
