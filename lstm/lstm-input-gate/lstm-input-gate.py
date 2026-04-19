import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    if len(h_prev.shape) == 1:
        i_t = sigmoid(np.matmul(W_i,np.concatenate((h_prev,x_t),axis = 0))+b_i)
        ct = np.tanh(np.matmul(W_c,np.concatenate((h_prev,x_t),axis = 0))+b_c)
    else:
        i_t = sigmoid(np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_i.T)+b_i)
        ct = np.tanh(np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_c.T)+b_c)
    return i_t,ct
    pass