import pandas as pd
import zipfile
from os import path
from sklearn.cross_validation import train_test_split
from numpy import matrix, array
from random import gauss, seed
from urllib import urlretrieve

DATA_ROOT = path.join(path.dirname(__file__), "..", "data") 
SPAMBASE_ROOT = path.join(DATA_ROOT, "spambase")


def parse(spambase_root, data_file="spambase.data"):
    df = pd.read_csv(
        path.join(spambase_root, data_file), 
        header=None, 
        names=header_names(spambase_root)
    )
    return df


def header_names(spambase_root, names_file="spambase.names"):
    names = []
    with open(path.join(spambase_root, names_file), "r") as f:
        for i, e in enumerate(f):
            if 32 < i < 90:
                names.append(e[:e.index(":")])
    names.append("spam")
    return names


def read_data(spambase_root=None, test_ratio=0.33, random_seed=1):
    if spambase_root is None:
        spambase_root = SPAMBASE_ROOT

    # get the data
    df = parse(spambase_root)
    ncol = len(df.columns)
  
    # split into training and test sets
    dftrain, dftest = train_test_split(df, 
        test_size=test_ratio, 
        random_state=random_seed) 
    
    # get and standardize the feature vectors 
    Xtrain = dftrain.ix[:, :(ncol - 1)]
    Xtrain = (Xtrain - Xtrain.mean()) / Xtrain.std()
    
    Xtest = dftest.ix[:, :(ncol - 1)]
    Xtest = (Xtest - Xtest.mean()) / Xtest.std()

    # get the class labels
    ytrain = dftrain.ix[:, (ncol - 1):]
    ytest = dftest.ix[:, (ncol - 1):]
        
    # add a constant column to simulate x0p0 being constant
    Xtrain.insert(0, "constant_term", 1) 
    Xtest.insert(0, "constant_term", 1)
    ncol += 1

    # convert to matrices
    Xtrain = matrix(Xtrain)
    Xtest = matrix(Xtest)
    ytrain = matrix(ytrain)
    ytest = matrix(ytest)
     
    # randomly initialize the parameter vector
    seed(random_seed)
    theta = array([gauss(0, 0.1) for i in range(ncol - 1)])

    return theta, Xtrain, Xtest, ytrain, ytest


def download(url=None, dest=None):
    if url is None:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "spambase/spambase.zip")
    if dest is None:
        dest = path.join(DATA_ROOT, "spambase.zip") 
    
    print("downloading spam data set ...")
    urlretrieve(url, dest)
    print("download complete, unzipping data ...") 
    with zipfile.ZipFile(dest) as z:
        z.extractall(SPAMBASE_ROOT) 
    print("complete")

