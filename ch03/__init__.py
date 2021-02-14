# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import pandas as pd
import os

# to make this notebook's output stable across runs
np.random.seed(42)

from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def run_ch03():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    print(mnist.keys())

    X, y = mnist["data"], mnist["target"]

    print(X.shape)
    print(y.shape)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    some_digit = X[0]
    some_digit_image = some_digit.reshape(28, 28)

    # plt.imshow(some_digit_image, cmap="binary")
    # plt.axis("off")
    # plt.show()
    print(y[0])
    y = y.astype(np.uint8)

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    print(sgd_clf.predict([some_digit]))
    from sklearn.model_selection import cross_val_score
    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

    never_5_clf = Never5Classifier()
    cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_train_5, y_train_pred))

    y_train_perfect_predictions = y_train_5
    print(confusion_matrix(y_train_5, y_train_perfect_predictions))

    from sklearn.metrics import precision_score, recall_score
    print(precision_score(y_train_5, y_train_pred))
    print(recall_score(y_train_5, y_train_pred))

    from sklearn.metrics import f1_score
    print(f1_score(y_train_5, y_train_pred))

    y_scores = sgd_clf.decision_function([some_digit])
    print(y_scores)
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)
    print(y_some_digit_pred)

    threshold = 8000
    y_some_digit_pred = (y_scores > threshold)
    print(y_some_digit_pred)

    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                 method="decision_function")

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.legend(loc="center right", fontsize=16)
        plt.xlabel("Threshold", fontsize=16)
        plt.grid(True)
        plt.axis([-50000, 50000, 0, 1])

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    y_train_pred_90 = (y_scores >= threshold_90_precision)
    precision_score(y_train_5, y_train_pred_90)
    recall_score(y_train_5, y_train_pred_90)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
        plt.axis([0, 1, 0, 1])  # Not shown in the book
        plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)  # Not shown
        plt.ylabel('True Positive Rate (Recall)', fontsize=16)  # Not shown
        plt.grid(True)

    plot_roc_curve(fpr, tpr)
    plt.show()

    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train_5, y_scores)

    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")

    y_scores_forest = y_probas_forest[:, 1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    plt.plot(fpr, tpr, "b:", label="sgd")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.svm import SVC

    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    svm_clf.predict([some_digit])

    some_digit_scores = svm_clf.decision_function([some_digit])

    np.argmax(some_digit_scores)
    print(svm_clf.classes_)
    print(svm_clf.classes_[5])

    from sklearn.multiclass import OneVsRestClassifier
    ovr_clf = OneVsRestClassifier(SVC())
    ovr_clf.fit(X_train, y_train)

    print(len(ovr_clf.estimators_))

    sgd_clf.fit(X_train, y_train)
    sgd_clf.predict([some_digit])

    sgd_clf.decision_function([some_digit])

    cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


