import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """

    n2 = np.linalg.norm(W_hh,2)
    a = 1
    l = []
    for t in range(T):
        l.append(a)
        a = a*n2
    return l
    
    pass