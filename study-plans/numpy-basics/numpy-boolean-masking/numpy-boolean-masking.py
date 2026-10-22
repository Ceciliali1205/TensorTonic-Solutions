import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    arr = np.array(data, dtype=np.float64)
    # element masking
    elem_mask = (arr > threshold).astype(np.float64)
    # any row
    any_row_mask = np.any(arr > threshold, axis=1)
    any_row = np.where(any_row_mask[:,np.newaxis], arr, 0.0)
    #all 
    all_row_mask = np.all(arr > threshold, axis = 1)
    all_row = np.where(all_row_mask[:, np.newaxis], arr, 0.0)
    return np.stack([elem_mask, any_row, all_row])
    
    
