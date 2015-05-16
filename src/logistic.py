from math import exp
from numpy import vectorize


def cost(X, y, theta):
    """Compute the cost for a given value of the training data and parameters.    
    
    :param X: An n x m  matrix containing the input data
    :type X: numpy.matrix
    :param y: A row vector of length m containing the predictions of the 
        training set
    :type y: numpy.matrix
    :param theta: A column vector of length n containing the model parameters
    :type theta: numpy.matrix
    """
    pass


def regularized_cost():
    pass


def predict(X, theta):
    """Create a vector of predictions.

    :param X: An n x m matrix containing the input data
    :type X: numpy.matrix
    :param theta: A column vector of length m containing the model parameters
    :type theta: numpy.matrix
    :rtype: numpy.matrix
    """
    return vectorize(sigmoid)(X*theta)


def sigmoid(x):
    """Calculate the sigmoid function for x.

    :param x: a number
    :type x: float
    :rtype: float
    """
    return 1.0 / (1.0 + exp(-1.0 * x))

