import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    weights = np.zeros((X.shape[1], num_classes), dtype=np.float64)
    bias = np.zeros(num_classes, dtype=np.float64)
    targets = np.eye(num_classes, dtype=np.float64)[y]
    for _ in range(n_iters):
        logits = X@weights + bias
        logits -= np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= np.sum(prob, axis=1, keepdims=True)
        error = prob - targets
        weights -= lr * ((X.T@error)/len(y))
        bias -= lr * np.mean(error, axis=0)
    return [[round(float(value), 4) for value in row] for row in weights], [round(float(value), 4) for value in bias]       
        
        
    
