import numpy as np

def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))