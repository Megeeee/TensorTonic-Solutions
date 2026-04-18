import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x=np.negative(x)
    return 1/(1+np.exp(x))
    pass