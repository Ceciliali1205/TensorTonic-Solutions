import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    # Convert X and y to numpy float64 arrays
    # Modifies/references the original without copying if the input is already a compatible array, faster and memory-optimized
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        y_pred = X@weights + bias * 1 
        error = y_pred - y
        weights -= lr * (2.0/len(y)) * X.T @ error
        bias -= lr * (2.0/len(y)) * np.sum(error)
    return weights, bias
        
        
    
