import numpy as np

from numpy.typing import ArrayLike, NDArray

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, gd_type: str = 'BGd', batches: int = 1) -> None:
        self.learning_rate: float = learning_rate
        self.n_iters: int = iterations
        self.gd_type: str = gd_type
        self.batches: int = batches
        self.weight: NDArray[np.float64] | np.float64 | None = None
        self.bias: np.float64 | None = None

    def __str__(self) -> str:
        # Extract: make attribute access robust and readable
        learning_rate = getattr(self, "learning_rate", 0.01)
        iterations = getattr(self, "n_iters", getattr(self, "iterations", 0))
        bias = getattr(self, "bias", 0.0)

        weight = getattr(self, "weight", None)
        if weight is None:
            w0 = 0.0
        else:
            try:
                # First coefficient if array-like
                w0 = float(weight[0])
            except Exception:
                # Scalar or non-indexable fallback
                try:
                    w0 = float(weight)
                except Exception:
                    w0 = 0.0

        # Clear, consistent string representation
        return f"α: [{learning_rate}] n: [{iterations}] w: [{w0}] b: [{bias}]"

    # Checks if valid arrays
    def _prepare_X_y_(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Normalize to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Ensure X is 2D: [n_samples, n_features], and Ensure y is 1D [n_samples,]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X.shape[0]}) does not match number of targets in y ({y.shape[0]})"
            )
        return X, y

    def _batchGradientDescent_(self, X: NDArray[np.float64], y: NDArray[np.float64], n_data):

        weight = self.weight
        bias = self.bias
        assert weight is not None and bias is not None

        # predictions and error
        y_hat = X @ self.weight + self.bias  # type: ignore
        error = y_hat - y

        # gradients
        derivative_w = (1.0 / n_data) * (X.T @ error)
        derivative_b = (1.0 / n_data) * np.sum(error)

        # parameter updates
        self.weight = weight - self.learning_rate * derivative_w
        self.bias = bias - self.learning_rate * derivative_b

    def _stochasticGradientDescent_(self, X: NDArray[np.float64], y: NDArray[np.float64], n_data):
        # Shuffling rows
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(int(n_data)):
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]
            # predictions and error
            y_hat_i = x_i @ self.weight + self.bias
            error = y_hat_i - y_i

            # gradients
            dw = x_i * error
            db = error  # type: ignore

            # parameter updates
            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    # Train data
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X_matrix, y_matrix = self._prepare_X_y_(X, y)
        n_data = float(X.shape[0])
        features = X.shape[1]

        if self.weight is None:
            self.weight = np.zeros((features,), dtype=np.float64)
        if self.bias is None:
            self.bias = np.float64(0.0)

        # Batch Gradient Descent
        if self.gd_type == 'BGd':
            if self.batches == 1:
                for _ in range(self.n_iters):
                    self._batchGradientDescent_(X_matrix, y_matrix, n_data)
            elif self.batches >= 2:
                n_rows = int(n_data / self.batches)
                for _ in range(self.n_iters):
                    # Shuffling rows
                    indices = np.random.permutation(len(X))
                    X_shuffled = X_matrix[indices]
                    y_shuffled = y_matrix[indices]

                    for i in range(self.batches):
                        # Separate into batch
                        X_batch = X_shuffled[(i * n_rows):n_rows]
                        y_batch = y_shuffled[(i * n_rows):n_rows]
                        self._batchGradientDescent_(X_batch, y_batch, n_rows)
            else:
                raise ValueError(f"Batch size {self.batches} not supported. Enter a batch size >= 1")


        # Stochastic Gradient Descent
        elif self.gd_type == 'SGd':
            for _ in range(self.n_iters):
                self._batchGradientDescent_(X_matrix, y_matrix, n_data)

        else:
            raise ValueError(f"GD: {self.gd_type} is not supported.")

    def predict(self, X) -> ArrayLike:

        if getattr(self, "weight", None) is None or getattr(self, "bias", None) is None:
            raise RuntimeError("Model is not trained yet. Call train(X, y) before predict(X).")

        X = np.asarray(X, dtype=np.float64)

        # Normalize Shapes
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim == 2 and X.shape[1] == 1:
            # X is already (n, 1), keep as is
            pass

        w = np.asarray(self.weight, dtype=float).reshape(-1)  # (m,)
        b = float(np.asarray(self.bias, dtype=float).reshape(()))  # scalar

        # If weight is scalar but X has one column, align dims
        if w.size == 1 and X.shape[1] != 1:
            raise ValueError(f"Model has 1 weight but X has {X.shape[1]} feature columns.")
        if w.size != 1 and X.shape[1] != w.size:
            raise ValueError(f"Feature mismatch: X has {X.shape[1]} columns, weight has {w.size}.")

        y = X @ w + b
        return np.asarray(y, dtype=float).reshape(-1)