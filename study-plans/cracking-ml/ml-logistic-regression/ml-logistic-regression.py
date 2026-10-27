import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(n_iters):
        z_hat = X@weights + bias
        prob = 1/(1+np.exp(-z_hat))
        error = prob - y
        weights -= lr * (1/len(y)) * (X.T @ error)
        bias -= lr * (1/len(y)) * np.sum(error)

    return weights, bias
