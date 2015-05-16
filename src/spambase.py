import pandas as pd
from os import path
from numpy import matrix, array
from random import gauss


def parse(data_root, data_file="spambase.data"):
    df = pd.read_csv(
        path.join(data_root, data_file), 
        header=None, 
        names=header_names(data_root)
    )
    return df


def header_names(data_root, names_file="spambase.names"):
    names = []
    with open(path.join(data_root, names_file), "r") as f:
        for i, e in enumerate(f):
            if 32 < i < 90:
                names.append(e[:e.index(":")])
    names.append("spam")
    return names


def get_data(data_root):
    df = parse(data_root)
    ncol = len(df.columns)
    
    # an initial guess of the parameter vector
    theta = array([gauss(0, 0.1) for i in range(ncol - 1)])

    # get and standardize the data matrix
    X = df.ix[:, :(ncol - 1)]
    X = (X - X.mean()) / X.std()
    X = matrix(X)

    # the predictions
    y = df.ix[:, (ncol - 1):]
    y = matrix(y)
    
    return theta, X, y

