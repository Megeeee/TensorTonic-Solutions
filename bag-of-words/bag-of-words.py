import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vector = np.zeros((len(vocab),),dtype = int)
    for token in tokens:
        if token in vocab:
            i = vocab.index(token)
            vector[i] += 1

    return vector
    pass