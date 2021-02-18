import numpy as np
from scipy.ndimage.interpolation import shift

np.random.seed(42)

def shift_image(image, dx, dy):
    image = image.reshage((28,28))
    shifted_image = shift(image, [dx, dy], cval=0, mode='constant')
    return shifted_image.reshape([-1])

def run_ex_0301():
    pass