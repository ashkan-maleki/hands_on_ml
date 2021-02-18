import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)


def run_ex_0301():
    X_test, X_train, y_test, y_train = fetch_mnist()

    param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    y_pred = grid_search.predict(X_test)
    print(accuracy_score(y_test, y_pred))


def fetch_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_test, X_train, y_test, y_train
