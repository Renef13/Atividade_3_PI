import numpy as np
import cv2 as cv
from scipy.ndimage import convolve

def det_roberts(img):
    # 1. Cinza + float64
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)
    # 2. Declarar kernel_gx e kernel_gy
    kernel_gx = np.array([[1, 0],
                      [0, -1]], dtype=np.float64)
    kernel_gy = np.array([[0, 1],
                      [-1, 0]], dtype=np.float64)
    # 3. gx = convolve(img_gray, kernel_gx, mode=?)
    gx = convolve(img_gray, kernel_gx, mode='reflect')
    # 4. gy = convolve(img_gray, kernel_gy, mode=?)
    gy = convolve(img_gray, kernel_gy, mode='reflect')
    # 5. magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # 6. Normalizar para [0, 255]
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # 7. Retornar como uint8
    return magnitude.astype(np.uint8)