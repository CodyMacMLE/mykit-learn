import numpy as np

from numpy.typing import ArrayLike, NDArray

# Checks if valid arrays
def _prepare_X_y(X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, gd_type: str = 'BGd', batches: int = 1) -> None:
        self.learning_rate: float = learning_rate
        self.n_iters: int = iterations
        self.gd_type: str = gd_type
        self.batches: int = batches
        self.weight: NDArray[np.float64] | np.float64 | None = None
        self.bias: np.float64 | None = None
        self.cost: float | None = None

    def __str__(self) -> str:
        # Extract: make attribute access robust and readable
        learning_rate = getattr(self, "learning_rate", 0.01)
        iterations = getattr(self, "n_iters", getattr(self, "iterations", 0))
        bias = getattr(self, "bias", 0.0)
        weight = getattr(self, "weight", None)

        # Clear, consistent string representation
        return f"α: [{learning_rate}] n: {iterations} w: {weight}  b: {bias}"

    def _calculateCost_(self, error, n_data) -> float:
        cost = (1.0 / (2 * n_data)) * np.sum(error ** 2)
        return cost

    def _batchGradientDescent_(self, X: NDArray[np.float64], y: NDArray[np.float64], n_data) -> float:

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

        cost = self._calculateCost_(error, n_data)

        return cost

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

    def _check_convergence_(self, cost: float) -> bool:
        epsilon = 0.000001

        if self.cost is None:
            self.cost = cost
            return False
        elif abs(self.cost - cost) < epsilon:
            return True
        else: # Update cost for next iteration
            self.cost = cost
            return False

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X_matrix, y_matrix = _prepare_X_y(X, y)
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
                    cost = self._batchGradientDescent_(X_matrix, y_matrix, n_data)
                    if self._check_convergence_(cost):
                        break
            elif self.batches >= 2:
                n_rows = int(n_data / self.batches)
                for _ in range(self.n_iters):
                    # Shuffling rows
                    indices = np.random.permutation(len(X))
                    X_shuffled = X_matrix[indices]
                    y_shuffled = y_matrix[indices]

                    for i in range(self.batches):
                        # Separate into batch
                        X_batch = X_shuffled[(i * n_rows):(i + 1) * n_rows]
                        y_batch = y_shuffled[(i * n_rows):(i + 1) * n_rows]
                        self._batchGradientDescent_(X_batch, y_batch, n_rows)

                    y_hat = X_matrix @ self.weight + self.bias
                    error = y_hat - y_matrix
                    cost = self._calculateCost_(error, n_data)

                    if self._check_convergence_(cost):
                        break

            else:
                raise ValueError(f"Batch size {self.batches} not supported. Enter a batch size >= 1")


        # Stochastic Gradient Descent
        elif self.gd_type == 'SGd':
            for _ in range(self.n_iters):
                self._stochasticGradientDescent_(X_matrix, y_matrix, n_data)
                y_hat = X_matrix @ self.weight + self.bias
                error = y_hat - y_matrix
                cost = self._calculateCost_(error, n_data)
                if self._check_convergence_(cost):
                    break

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

        w = np.asarray(self.weight, dtype=float)  # (m,)
        b = float(np.asarray(self.bias, dtype=float).reshape(()))  # scalar

        # If weight is scalar but X has one column, align dims
        if w.size == 1 and X.shape[1] != 1:
            raise ValueError(f"Model has 1 weight but X has {X.shape[1]} feature columns.")
        if w.size != 1 and X.shape[1] != w.size:
            raise ValueError(f"Feature mismatch: X has {X.shape[1]} columns, weight has {w.size}.")

        y = X @ w + b

        return np.asarray(y, dtype=float).reshape(-1)

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, iterations: int = 1000, gd_type: str = 'BGd', batches: int = 1) -> None:
        self.learning_rate: float = learning_rate
        self.n_iters: int = iterations
        self.gd_type: str = gd_type
        self.batches: int = batches
        self.weight: NDArray[np.float64] | np.float64 | None = None
        self.bias: np.float64 | None = None
        self.cost: float | None = None

    def __str__(self) -> str:
        return "Not implemented."

    def error(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        z = X @ self.weight + self.bias
        predictions = self._sigmoid(z)
        return np.mean((predictions - y) ** 2)

    @staticmethod
    def _sigmoid(z):
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _calculateCost_(self, n_data, y_hat, y) -> float:
        y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
        cost = -(1.0 / n_data) * (y.T @ np.log(y_hat_clipped) + (1-y).T @ np.log(1-y_hat_clipped))
        return cost

    def _check_convergence_(self, cost: float) -> bool:
        epsilon = 0.000001

        if self.cost is None:
            self.cost = cost
            return False
        elif abs(self.cost - cost) < epsilon:
            return True
        else:  # Update cost for next iteration
            self.cost = cost
            return False

    def _batchGradientDescent_(self, X: NDArray[np.float64], y: NDArray[np.float64], n_data) -> float:
        weight = self.weight
        bias = self.bias
        assert weight is not None and bias is not None

        # predictions using sigmoid function
        z = X @ weight + bias
        y_hat = self._sigmoid(z)
        error = y_hat - y

        # gradients
        derivative_w = (1.0 / n_data) * (X.T @ error)
        derivative_b = (1.0 / n_data) * np.sum(error)

        # parameter updates
        self.weight = weight - self.learning_rate * derivative_w
        self.bias = bias - self.learning_rate * derivative_b

        cost = self._calculateCost_(n_data, y_hat, y)

        return cost

    def _stochasticGradientDescent_(self, X, y, n_data) -> float:
        # Shuffling rows
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(int(n_data)):
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]

            # predictions and error
            z = x_i @ self.weight + self.bias
            y_hat_i = self._sigmoid(z)
            error = y_hat_i - y_i

            # gradients
            derivative_w = x_i * error
            derivative_b = error  # type: ignore

            # parameter updates
            self.weight = self.weight - self.learning_rate * derivative_w
            self.bias = self.bias - self.learning_rate * derivative_b

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X_matrix, y_matrix = _prepare_X_y(X, y)
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
                    cost = self._batchGradientDescent_(X_matrix, y_matrix, n_data)
                    if self._check_convergence_(cost):
                        break
            # Mini-Batch
            elif self.batches >= 2:
                n_rows = int(n_data / self.batches)
                for _ in range(self.n_iters):
                    # Shuffling rows
                    indices = np.random.permutation(len(X))
                    X_shuffled = X_matrix[indices]
                    y_shuffled = y_matrix[indices]

                    for i in range(self.batches):
                        # Separate into batch
                        X_batch = X_shuffled[(i * n_rows):(i + 1) * n_rows]
                        y_batch = y_shuffled[(i * n_rows):(i + 1) * n_rows]
                        self._batchGradientDescent_(X_batch, y_batch, n_rows)

                    z = X_matrix @ self.weight + self.bias
                    y_hat = self._sigmoid(z)
                    cost = self._calculateCost_(n_data, y_hat, y_matrix)

                    if self._check_convergence_(cost):
                        break

            else:
                raise ValueError(f"Batch size {self.batches} not supported. Enter a batch size >= 1")


        # Stochastic Gradient Descent
        elif self.gd_type == 'SGd':
            for _ in range(self.n_iters):
                self._stochasticGradientDescent_(X_matrix, y_matrix, n_data)
                z = X_matrix @ self.weight + self.bias
                y_hat = self._sigmoid(z)
                cost = self._calculateCost_(n_data, y_hat, y_matrix)
                if self._check_convergence_(cost):
                    break

        else:
            raise ValueError(f"GD: {self.gd_type} is not supported.")

    def predict(self, X: NDArray[np.float64], proba: bool = True) -> NDArray[np.float64]:
        z = X @ self.weight + self.bias
        predictions = self._sigmoid(z)
        return np.asarray(predictions, dtype=float).reshape(-1) if proba else (predictions>=0.5).astype(np.int64)