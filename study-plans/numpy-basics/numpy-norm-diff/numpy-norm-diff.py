import numpy as np

def norm_diff(a: list, b: list, lo: float, hi: float) -> np.ndarray:
    """
    Returns a float64 array of absolute normalized differences.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a_c = np.clip(a, lo, hi)
    b_c = np.clip(b, lo, hi)
    return np.abs(a_c - b_c)/(hi - lo)
