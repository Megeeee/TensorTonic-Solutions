import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    if len(h_prev.shape) == 1:
        f = np.matmul(W_f,np.concatenate((h_prev,x_t),axis = 0))+b_f
    else:
        f = np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_f.T)+b_f
    return sigmoid(f)
    pass