from data_genration import genrate_regression_data 
from gradient_decent_numpy import gradient_decent
from visualizations import ( plot_loss, plot_regression,plot_lr_comparsion,plot_numpy_vs_sklearn)
from sklearn_implementaion import train_skearn_model

X,y = genrate_regression_data()

w,b,losses = gradient_decent(X,y,lr=0.01)
plot_loss(losses,"MSE Loss (Numpy Gradient decent)")
plot_regression(X,y,w,b)

plot_lr_comparsion(X,y,[0.001,0.01,0.1],gradient_decent)

sk_model = train_skearn_model(X,y)

plot_numpy_vs_sklearn(X,y,w,b,sk_model)