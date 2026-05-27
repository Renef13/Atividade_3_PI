import numpy as np
import cv2 as cv
from scipy.ndimage import convolve

def det_roberts(img):
    # 1. Cinza + float64
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float64)
    # 2. Declarar kernel_gx e kernel_gy
    kernel_gx = []
    # 3. gx = convolve(img_gray, kernel_gx, mode=?)
    # 4. gy = convolve(img_gray, kernel_gy, mode=?)
    # 5. magnitude = np.sqrt(gx**2 + gy**2)
    # 6. Normalizar para [0, 255]
    # 7. Retornar como uint8
    pass