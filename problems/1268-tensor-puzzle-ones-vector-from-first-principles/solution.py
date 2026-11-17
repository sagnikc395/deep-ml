import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    # Your code here
    ones = (np.arange(n) * 0) + np.array([1.0] ,dtype=np.float64)
    return ones 

