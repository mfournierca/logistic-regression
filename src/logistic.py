from math import exp
from numpy import vectorize, ones, matrix, log
from scipy import optimize


def cost(theta, X, y):
    """Compute the cost for a given value of the training data and parameters.    
    
    :param theta: A 1-D vector of length m containing the model parameters
    :type theta: numpy.array
    :param X: An n x m  matrix containing the input data
    :type X: numpy.matrix
    :param y: A row vector of length n containing the predictions of the 
        training set
    :type y: numpy.matrix
    :rtype: float
    """
    
    n, m = X.shape
    theta = matrix(theta).transpose()
 
    assert n == y.shape[0], "number of rows in X does not match y"
    assert m == theta.shape[0], "number of columns in X does not match theta"
    
    j = (1.0 / float(n)) * (
        matrix(-1.0) * y.transpose() * log(predict(theta, X)) -
        (ones((1, n)) - y.transpose()) * log(ones((n, 1)) - predict(theta, X))
    )
    return float(j)


def regularized_cost():
    pass


def gradient():
    pass


def predict(theta, X):
    """Create a vector of predictions.

    :param theta: A column vector of length m containing the model parameters
    :type theta: numpy.matrix
    :param X: An n x m matrix containing the input data
    :type X: numpy.matrix
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


def optimize(theta, X, y):
    pass

