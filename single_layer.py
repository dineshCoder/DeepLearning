import numpy as np
import logging
from activation_functions import sigmoid, reLU
logger = logging.getLogger(__name__)

def single_neutron(W, X, activation_function=reLU):
    """
    This is a model of single neuron. It takes Weights vector
    and Input vector. Takes their dot product and passes through
    a non-linearity and gives the output.
    :param W: Weight Vector
    :param X: Input Vector
    :return: a single real number
    """
    assert isinstance(W, np.ndarray) == True, "W is expected to be a numpy array."
    assert isinstance(X, np.ndarray) == True, "X is expected to be a numpy array."
    shape_w = W.shape
    shape_x = X.shape
    assert (shape_x[0]==shape_w[0] and len(shape_w)==1 and len(shape_x)==1) or (len(shape_x)==1 and len(shape_w)==2 and shape_w[1]==1 and shape_x[0]==shape_w[0]) or (len(shape_w)==1 and len(shape_x)==2 and shape_x[1]==1 and shape_x[0]==shape_w[0]), "Shape compatibility problem"
    temp = np.dot(W.T,X)
    return activation_function(temp)

if __name__ == "__main__":
    X = np.array([0,-2,1.98])
    W = np.array([1,2,1])
    print(single_neutron(W,X))

