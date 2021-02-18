import numpy as np
from scipy.ndimage.interpolation import shift
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from .ex0301_kmeans import fetch_mnist

np.random.seed(42)


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


def test_shift_image(X_train):
    image = X_train[1000]
    shifted_image_down = shift_image(image, 0, 5)
    shifted_image_left = shift_image(image, -5, 0)

    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.title("Original", fontsize=14)
    plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
    plt.subplot(132)
    plt.title("Shifted down", fontsize=14)
    plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
    plt.subplot(133)
    plt.title("Shifted left", fontsize=14)
    plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
    plt.show()


def augment_images(X_train, y_train):
    X_train_augmented = [image for image in X_train]
    y_train_augmented = [label for label in y_train]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)

    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    return X_train_augmented, y_train_augmented


def run_ex_0301():
    X_test, X_train, y_test, y_train = fetch_mnist()
    test_shift_image(X_train)
    X_train_augmented, y_train_augmented = augment_images(X_train, y_train)

    X_train_augmented, y_train_augmented = shuffle_data(X_train_augmented, y_train_augmented)

    train_knn(X_test, X_train_augmented, y_test, y_train_augmented)


def train_knn(X_test, X_train_augmented, y_test, y_train_augmented):
    knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
    knn_clf.fit(X_train_augmented, y_train_augmented)
    y_pred = knn_clf.predict(X_test)
    accuracy_score(y_test, y_pred)


def shuffle_data(X_train_augmented, y_train_augmented):
    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_idx]
    y_train_augmented = y_train_augmented[shuffle_idx]
    return X_train_augmented, y_train_augmented
