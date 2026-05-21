import cv2
import numpy as np


def average_dithering(img_bgr):
    img_gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float64)

    h, w = img_gray.shape
    out = np.zeros_like(img_gray)

    threshold = 127

    for i in range(h):
        for j in range(w):
            pixel = img_gray[i, j]
            out[i, j] = 255.0 if pixel >= threshold else 0.0

    out_uint8 = out.astype(np.uint8)

    img_rgb = cv2.cvtColor(
        out_uint8,
        cv2.COLOR_GRAY2RGB
    )

    return img_rgb


def random_dithering(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    h, w = img_gray.shape
    out = np.zeros_like(img_gray)

    for i in range(h):
        for j in range(w):
            pixel = img_gray[i, j]
            threshold = np.random.uniform(0.0, 255.0)
            out[i, j] = 255.0 if pixel >= threshold else 0.0

    out_uint8 = out.astype(np.uint8)

    img_rgb = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)

    return img_rgb


def ordered_dithering_4x4(img_bgr):
    BAYER_4x4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float64)

    BAYER_4x4 = (BAYER_4x4 / 16.0) * 255.0

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    h, w = img_gray.shape
    out = np.zeros_like(img_gray)

    for i in range(h):
        for j in range(w):
            pixel = img_gray[i, j]
            threshold = BAYER_4x4[i % 4, j % 4]
            out[i, j] = 255.0 if pixel >= threshold else 0.0

    out_uint8 = out.astype(np.uint8)

    img_rgb = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)

    return img_rgb


def ordered_dithering_8x8(img_bgr):
    BAYER_8x8 = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float64)

    BAYER_8x8 = (BAYER_8x8 / 64.0) * 255.0

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    h, w = img_gray.shape
    out = np.zeros_like(img_gray)

    for i in range(h):
        for j in range(w):
            pixel = img_gray[i, j]
            threshold = BAYER_8x8[i % 8, j % 8]
            out[i, j] = 255.0 if pixel >= threshold else 0.0

    out_uint8 = out.astype(np.uint8)

    img_rgb = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)

    return img_rgb


def floyd_steinberg(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    h, w = img_gray.shape
    out = img_gray.copy()

    for i in range(h):
        for j in range(w):
            pixel_antigo = out[i, j]
            pixel_novo = 255.0 if pixel_antigo >= 127.0 else 0.0
            erro = pixel_antigo - pixel_novo

            out[i, j] = pixel_novo

            if j + 1 < w:
                out[i, j + 1] += erro * (7 / 16)
            if i + 1 < h and j - 1 >= 0:
                out[i + 1, j - 1] += erro * (3 / 16)
            if i + 1 < h:
                out[i + 1, j] += erro * (5 / 16)
            if i + 1 < h and j + 1 < w:
                out[i + 1, j + 1] += erro * (1 / 16)

    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)

    return img_rgb
