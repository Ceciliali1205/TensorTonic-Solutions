import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    a=np.array(data,dtype=np.float64)
    original=a[row_idx]
    return np.stack([original, np.clip(original, lo, hi)])
