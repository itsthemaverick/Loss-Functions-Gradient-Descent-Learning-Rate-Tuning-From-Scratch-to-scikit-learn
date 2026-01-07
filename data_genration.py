import numpy as np

def genrate_regression_data(n=200):
    np.random.seed(42)
    X = np.random.rand(n,1)*10
    y = 4*X.squeeze() + 7 + np.random.rand(n)*3
    return X,y