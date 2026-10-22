import numpy as np

def scale_cols(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each column multiplied by its weight.
    """
    arr = np.array(data, dtype=np.float64)
    weight = np.array(weights, dtype=np.float64)
    return arr*weights
