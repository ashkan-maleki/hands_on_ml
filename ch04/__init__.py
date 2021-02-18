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