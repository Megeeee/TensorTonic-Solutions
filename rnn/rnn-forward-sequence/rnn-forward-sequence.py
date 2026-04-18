import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch,T,dim = X.shape
    hidden_states = np.zeros((batch,T,h_0.shape[1]))
    for t in range(T):  
        h_0 = np.tanh(np.matmul(X[:,t,:],W_xh.T) + np.matmul(h_0,W_hh.T) + b_h)
        hidden_states[:,t,:] = h_0
    return hidden_states,h_0
    pass

