"""
Logistic Regression From Scratch
=================================
A pure NumPy implementation of Logistic Regression designed for binary
classification with heavy class imbalance.

Features:
- Numerically stable Sigmoid with exponential clipping
- Weighted Binary Cross-Entropy Loss (supporting 'balanced' class weights)
- Analytical Gradient Descent with L2 (Ridge) Regularization
- Dynamic loss tracking and convergence monitoring
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Logistic Regression classifier trained using Batch Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size taken in the direction of the negative gradient.
    n_iterations : int, default=1000
        Maximum number of gradient descent iterations.
    l2_penalty : float, default=0.01
        L2 regularization parameter (lambda) to prevent overfitting.
    class_weight : str, dict, or None, default='balanced'
        Weight applied to each class. If 'balanced', weights are inversely
        proportional to class frequencies: N / (2 * N_c).
    tolerance : float, default=1e-6
        Stopping criterion. If the absolute loss difference between consecutive
        iterations is less than tolerance, optimization terminates early.
    random_state : int or None, default=42
        Seed used to initialize weights randomly.
    verbose : bool, default=False
        If True, prints loss at regular iteration intervals.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        l2_penalty: float = 0.01,
        class_weight: str = "balanced",
        tolerance: float = 1e-6,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.class_weight = class_weight
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose

        # Model parameters
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.n_features_in_ = None
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Compute numerically stable sigmoid activation.
        Clips z to [-250, 250] to avoid float64 overflow in exp(-z).
        """
        z_clipped = np.clip(z, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute per-sample weight based on class_weight specification.
        """
        n_samples = len(y)
        if self.class_weight is None:
            return np.ones(n_samples, dtype=np.float64)

        if self.class_weight == "balanced":
            n_0 = np.sum(y == 0)
            n_1 = np.sum(y == 1)
            w0 = n_samples / (2.0 * max(n_0, 1))
            w1 = n_samples / (2.0 * max(n_1, 1))
            weights = np.where(y == 1, w1, w0).astype(np.float64)
            return weights

        if isinstance(self.class_weight, dict):
            w0 = float(self.class_weight.get(0, 1.0))
            w1 = float(self.class_weight.get(1, 1.0))
            return np.where(y == 1, w1, w0).astype(np.float64)

        raise ValueError(f"Unknown class_weight option: {self.class_weight}")

    def _compute_loss(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, sample_weights: np.ndarray
    ) -> float:
        """
        Compute weighted Binary Cross-Entropy loss with L2 regularization.
        """
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1.0 - eps)

        # Weighted log-loss
        bce = -sample_weights * (y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        mean_bce = np.sum(bce) / np.sum(sample_weights)

        # L2 penalty on weights (excluding bias)
        l2_reg = 0.5 * self.l2_penalty * np.sum(self.weights ** 2)
        return float(mean_bce + l2_reg)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit logistic regression model using gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Binary class labels (0 or 1).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Initialize weights with Xavier / small Gaussian initialization
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=n_features)
        else:
            self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        sample_weights = self._compute_sample_weights(y)
        total_sample_weight = np.sum(sample_weights)

        self.loss_history = []
        prev_loss = float("inf")

        for iteration in range(self.n_iterations):
            # Forward pass: linear combination + sigmoid activation
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)

            # Compute current loss
            current_loss = self._compute_loss(y, y_hat, sample_weights)
            self.loss_history.append(current_loss)

            # Check convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}: loss={current_loss:.6f}")
                break
            prev_loss = current_loss

            # Backward pass: Analytical gradients
            # error = y_hat - y, weighted by sample_weights
            weighted_error = sample_weights * (y_hat - y)

            # dJ/dw = (1 / total_weight) * (X^T @ weighted_error) + l2_penalty * w
            dw = (np.dot(X.T, weighted_error) / total_sample_weight) + (self.l2_penalty * self.weights)
            # dJ/db = (1 / total_weight) * sum(weighted_error)
            db = np.sum(weighted_error) / total_sample_weight

            # Gradient update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (iteration % (max(1, self.n_iterations // 10)) == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration:5d}/{self.n_iterations} | Loss: {current_loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates for each sample.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, 2)
            Column 0 contains P(y=0), Column 1 contains P(y=1).
        """
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.weights) + self.bias
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for samples using decision threshold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        threshold : float, default=0.5
            Decision boundary cutoff. If P(y=1) >= threshold, predicts 1.
        """
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy on given test data and labels.
        """
        preds = self.predict(X, threshold=0.5)
        return float(np.mean(preds == np.asarray(y).ravel()))
