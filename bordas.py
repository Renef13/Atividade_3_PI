import numpy as np
import cv2 as cv
from scipy.ndimage import convolve


def det_roberts(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)

    kernel_gx = np.array([[1, 0],
                          [0, -1]], dtype=np.float64)
    kernel_gy = np.array([[0, 1],
                          [-1, 0]], dtype=np.float64)

    gx = convolve(img_gray, kernel_gx, mode='reflect')

    gy = convolve(img_gray, kernel_gy, mode='reflect')

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    return magnitude.astype(np.uint8)


def det_prewitt(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    kernel_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    gx = convolve(img_gray, kernel_gx, mode='reflect')
    gy = convolve(img_gray, kernel_gy, mode='reflect')

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    return magnitude.astype(np.uint8)
