import numpy as np
import cv2 as cv
from scipy.ndimage import convolve, correlate
from filtros import filtro_gaussiana, gerar_kernel_gaussiano


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


def det_sobel(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)

    kernel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = cv.filter2D(img_gray, cv.CV_32F, kernel_gx)
    gy = cv.filter2D(img_gray, cv.CV_32F, kernel_gy)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    return magnitude.astype(np.uint8)


def det_scharr(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)

    kernel_gx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
    kernel_gy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64)

    gx = correlate(img_gray, kernel_gx, mode='reflect')
    gy = correlate(img_gray, kernel_gy, mode='reflect')

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    return magnitude.astype(np.uint8)


def det_laplaciano(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    lap = convolve(img_gray, kernel, mode='reflect')

    lap = np.abs(lap)

    lap = cv.normalize(lap, None, 0, 255, cv.NORM_MINMAX)

    return lap.astype(np.uint8)


def det_LoG(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)

    img_blur = convolve(
        img_gray,
        gerar_kernel_gaussiano(5),
        mode='reflect'
    )

    kernel_lap = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    log = convolve(
        img_blur,
        kernel_lap,
        mode='reflect'
    )

    log = np.abs(log)

    log = cv.normalize(
        log,
        None,
        0,
        255,
        cv.NORM_MINMAX
    )

    return log.astype(np.uint8)

def det_DoG(img):
    g1 = filtro_gaussiana(img, 3)
    g2 = filtro_gaussiana(img, 7)

    dog = cv.cvtColor(g1, cv.COLOR_BGR2GRAY).astype(np.float64) - \
          cv.cvtColor(g2, cv.COLOR_BGR2GRAY).astype(np.float64)

    dog = np.abs(dog)

    dog = cv.normalize(
        dog,
        None,
        0,
        255,
        cv.NORM_MINMAX
    )

    return dog.astype(np.uint8)

def det_canny(img):
    pass
