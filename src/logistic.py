from math import exp
from numpy import vectorize


def cost(X, y, theta):
    pass


def predict(X, theta):
    """Create a vector of predictions.

    :param X: a matrix containing the input data
    :type X: numpy.matrix
    :param theta: a column vector containing the model parameters
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


def regularized_cost():
    pass
