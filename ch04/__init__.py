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