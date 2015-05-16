from math import exp

def cost():
    pass


def sigmoid(x):
    """Calculate the sigmoid function for x. 

    :param x: a number
    :type x: float
    :rtype: float
    """
    return 1.0 / (1.0 + exp(-1.0 * x))


def regularized_cost():
    pass
