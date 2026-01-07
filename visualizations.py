import matplotlib.pyplot as plt
import numpy as np 

def plot_loss(losses,title="Loss vs epochs"):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.show()

def plot_regression(X,y,w,b):
    plt.figure()
    plt.scatter(X,y)
    x_line = np.linspace(X.min(),X.max(),100)
    y_line = w*x_line + b
    plt.plot(x_line,y_line)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regression Line")
    plt.show()

def plot_lr_comparsion(X,y,lrs,gradient_decent_fn):
    plt.figure()
    for lr in lrs:
        _,_,losses = gradient_decent_fn(X,y,lr=lr)
        plt.plot(losses,label=f"lr={lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Rate Comparasion")
        plt.show()

def plot_numpy_vs_sklearn(X,y,w_numpy,b_numpy,sklearn_model):
    plt.figure()
    plt.scatter(X,y,label="Data")
    x_line = np.linspace(X.min(),X.max(),100)

    y_numpy = w_numpy*x_line + b_numpy
    plt.plot(x_line,y_numpy,label="numpy gradient decent")

    y_sklearn = sklearn_model.predict(x_line.reshape(-1,1))
    plt.plot(x_line,y_sklearn,label="scikit-learn SDG")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(" Numpy vs Sklearn Regression Line ")
    plt.legend()
    plt.show()