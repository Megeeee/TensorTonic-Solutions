import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    if len(h_prev.shape) == 1:
        f_t = sigmoid(np.matmul(W_f,np.concatenate((h_prev,x_t),axis = 0))+b_f)
        i_t = sigmoid(np.matmul(W_i,np.concatenate((h_prev,x_t),axis = 0))+b_i)
        c_tilde = np.tanh(np.matmul(W_c,np.concatenate((h_prev,x_t),axis = 0))+b_c)
        o_t = sigmoid(np.matmul(W_o,np.concatenate((h_prev,x_t),axis = 0))+b_o)
    else:
        f_t = sigmoid(np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_f.T)+b_f)
        i_t = sigmoid(np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_i.T)+b_i)
        c_tilde = np.tanh(np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_c.T)+b_c)
        o_t = sigmoid(np.matmul(np.concatenate((h_prev,x_t),axis = 1),W_o.T)+b_o)
    
    c_t = f_t*C_prev+i_t*c_tilde
    h_t = o_t*np.tanh(c_t)
    return h_t,c_t
    
    pass