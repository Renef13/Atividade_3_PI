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

def det_canny(img, low_threshold=50, high_threshold=150):

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)
    kernel_gauss = gerar_kernel_gaussiano(5)
    img_suave = convolve(img_gray, kernel_gauss, mode='reflect')

    kernel_gx = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)
    kernel_gy = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)

    gx = convolve(img_suave, kernel_gx, mode='reflect')
    gy = convolve(img_suave, kernel_gy, mode='reflect')

    magnitude = np.sqrt(gx**2 + gy**2)

    angulo = np.arctan2(gy, gx) * 180.0 / np.pi
    angulo[angulo < 0] += 180.0


    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            theta = angulo[i, j]

            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif 22.5 <= theta < 67.5:
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= theta < 112.5:
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            else:
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]


            nms[i, j] = magnitude[i, j] if (magnitude[i, j] >= q and magnitude[i, j] >= r) else 0.0


    nms_norm = cv.normalize(nms, None, 0, 255, cv.NORM_MINMAX)

    FORTE = 255.0
    FRACO = 75.0

    resultado = np.zeros((h, w), dtype=np.float64)
    resultado[nms_norm >= high_threshold] = FORTE
    resultado[(nms_norm >= low_threshold) & (nms_norm < high_threshold)] = FRACO

    kernel_viz = np.ones((3, 3), dtype=np.float64)

    alterado = True
    while alterado:
        anterior  = resultado.copy()
        mapa_forte = (resultado == FORTE).astype(np.float64)
        expansao   = convolve(mapa_forte, kernel_viz, mode='constant', cval=0.0)
        resultado[(resultado == FRACO) & (expansao > 0)] = FORTE
        alterado = not np.array_equal(anterior, resultado)

    resultado[resultado == FRACO] = 0.0

    return resultado.astype(np.uint8)
