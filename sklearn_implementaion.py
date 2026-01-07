from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_skearn_model(X,y):
    model = Pipeline([
        ("scalar",StandardScaler()),
        ("sdg",SGDRegressor(
            loss="squared_error",
            learning_rate="constant",
            eta0 = 0.01,
            max_iter=1000,
            random_state=42
        ))
    ])
    model.fit(X,y)
    return model