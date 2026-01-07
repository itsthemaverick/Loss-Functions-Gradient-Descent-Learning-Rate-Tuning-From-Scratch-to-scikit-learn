import numpy as np

def mse(y,y_hat):
    return np.mean((y-y_hat)**2)

def mae(y,y_hat):
    return np.mean(np.abs(y-y_hat))

def log_loss(y,y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat,eps,1-eps)
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))