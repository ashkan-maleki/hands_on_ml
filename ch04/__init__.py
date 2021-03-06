import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


def run_ch04():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.rand(100, 1)

    X_b = np.c_[np.ones((100,1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(theta_best)

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2,1)), X_new]
    y_predict = X_new_b.dot(theta_best)
    print(y_predict)

    plt.plot(X_new, y_predict, 'r-')
    plt.plot(X, y, 'b.')
    plt.axis([0, 2, 0, 15])
    plt.show()

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    print(lin_reg.intercept_, lin_reg.coef_)
    print(lin_reg.predict(X_new))

    theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
    print(theta_best_svd)

    print(np.linalg.pinv(X_b).dot(y))

    eta = 0.1  # learning rate
    n_iterations = 1000
    m = 100

    theta = np.random.randn(2, 1)  # random initialization

    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients

    print(theta)

    m = len(X_b)

    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters

    def learning_schedule(t):
        return t0 / (t + t1)

    theta = np.random.randn(2, 1)  # random initialization

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients

    from sklearn.linear_model import SGDRegressor

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())

    print(sgd_reg.intercept_, sgd_reg.coef_)

    import numpy.random as rnd

    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    # save_fig("quadratic_data_plot")
    plt.show()

    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    print(X[0])

    print(X_poly[0])

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    def plot_learning_curves(model, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
        train_errors, val_errors = [], []
        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))

        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
        plt.legend(loc="upper right", fontsize=14)  # not shown in the book
        plt.xlabel("Training set size", fontsize=14)  # not shown
        plt.ylabel("RMSE", fontsize=14)  # not shown

    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    plt.axis([0, 80, 0, 3])  # not shown in the book
    # save_fig("underfitting_learning_curves_plot")  # not shown
    plt.show()  # not shown

    from sklearn.pipeline import Pipeline

    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

    plot_learning_curves(polynomial_regression, X, y)
    plt.axis([0, 80, 0, 3])  # not shown
    # save_fig("learning_curves_plot")  # not shown
    plt.show()  # not shown

    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)

    from sklearn.linear_model import Ridge
    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
    ridge_reg.fit(X, y)
    ridge_reg.predict([[1.5]])

    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
    ridge_reg.fit(X, y)
    ridge_reg.predict([[1.5]])

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    def plot_model(model_class, polynomial, alphas, **model_kargs):
        for alpha, style in zip(alphas, ("b-", "g--", "r:")):
            model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
            if polynomial:
                model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
            model.fit(X, y)
            y_new_regul = model.predict(X_new)
            lw = 2 if alpha > 0 else 1
            plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
        plt.plot(X, y, "b.", linewidth=3)
        plt.legend(loc="upper left", fontsize=15)
        plt.xlabel("$x_1$", fontsize=18)
        plt.axis([0, 3, 0, 4])

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(Ridge, polynomial=True, alphas=(0, 10 ** -5, 1), random_state=42)

    # save_fig("ridge_regression_plot")
    plt.show()

    sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
    sgd_reg.fit(X, y.ravel())
    sgd_reg.predict([[1.5]])

    from sklearn.linear_model import Lasso
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    lasso_reg.predict([[1.5]])

    from sklearn.linear_model import ElasticNet
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X, y)
    elastic_net.predict([[1.5]])

    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 2 + X + 0.5 * X ** 2 + np.random.randn(m, 1)

    X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

    from sklearn.base import clone

    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])

    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.transform(X_val)

    sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                           penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)

    from sklearn import datasets
    iris = datasets.load_iris()
    list(iris.keys())

    print(iris.DESCR)

    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver="lbfgs", random_state=42)
    log_reg.fit(X, y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)

    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

    plt.figure(figsize=(8, 3))
    plt.plot(X[y == 0], y[y == 0], "bs")
    plt.plot(X[y == 1], y[y == 1], "g^")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
    plt.text(decision_boundary + 0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])
    # save_fig("logistic_regression_plot")
    plt.show()

    print(decision_boundary)
    print(log_reg.predict([[1.7], [1.5]]))

    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)

    print(softmax_reg.predict([[5, 2]]))

    print(softmax_reg.predict_proba([[5, 2]]))