import numpy as np

def quantize_and_frame(data: list, decimals: int, pad_width: int) -> np.ndarray:
    """
    Returns float64 slices of rounded, floored, and ceiling-rounded values with zero borders.
    """
    a = np.array(data, dtype=np.float64)
    pad = lambda x: np.pad(x, pad_width, mode="constant", constant_values=0.0)
    rounded = pad(np.round(a, decimals))
    floored = pad(np.floor(a))
    ceiled = pad(np.ceil(a))
    return np.stack([rounded, floored, ceiled])
