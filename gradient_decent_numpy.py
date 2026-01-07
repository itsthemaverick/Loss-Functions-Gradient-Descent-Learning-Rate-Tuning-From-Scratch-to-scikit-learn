import numpy as np 
from loss_functions import mse

def gradient_decent(X,y,lr=0.01,epochs=1000):
    w,b = 0.0,0.0
    n = len(X)
    losses = []

    X_1d = X.squeeze()

    for _ in range(epochs):
        y_hat = w*X_1d + b
        error = y-y_hat

        dw = (-2/n)*np.sum(X_1d*error)
        db = (-2/n)*np.sum(error) 

        w = lr*dw
        b = lr*db

        losses.append(mse(y,y_hat))

    return w,b,losses