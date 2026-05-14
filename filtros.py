import numpy as np
import cv2
from scipy.ndimage import convolve


def filtro_media(img, tamanho_kernel=3):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w, c = img_lab.shape
    out = np.zeros_like(img_lab)
    margem = tamanho_kernel // 2

    img_pad = np.pad(img_lab, ((margem, margem), (margem, margem), (0, 0)), mode='edge')

    for i in range(h):
        for j in range(w):
            janela = img_pad[i:i + tamanho_kernel, j:j + tamanho_kernel, :]
            media = janela.mean(axis=(0, 1))
            out[i, j] = media

    out_lab = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def filtro_mediana(img, tamanho_kernel=3):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w, c = img_lab.shape
    out = np.zeros_like(img_lab)
    margem = tamanho_kernel // 2

    img_pad = np.pad(img_lab, ((margem, margem), (margem, margem), (0, 0)), mode='edge')

    for i in range(h):
        for j in range(w):
            janela = img_pad[i:i + tamanho_kernel, j:j + tamanho_kernel, :]
            mediana = np.median(janela, axis=(0, 1))
            out[i, j] = mediana

    out_lab = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def gerar_kernel_gaussiano(tamanho):
    if tamanho % 2 == 0:
        tamanho += 1

    sigma = (tamanho - 1) / 6

    centro = tamanho // 2

    x, y = np.mgrid[
        -centro:centro + 1,
        -centro:centro + 1
    ]

    kernel = np.exp(
        -((x ** 2 + y ** 2) / (2 * sigma ** 2))
    )

    kernel /= np.sum(kernel)

    return kernel


def filtro_gaussiana(img_bgr, tamanho_kernel=3):
    img_lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float64)

    l, a, b = cv2.split(img_lab)

    kernel = gerar_kernel_gaussiano(
        tamanho_kernel
    )

    l_filtrado = convolve(
        l,
        kernel,
        mode='reflect'
    )

    img_lab_filtrada = cv2.merge([
        l_filtrado,
        a,
        b
    ])

    img_bgr_filtrada = cv2.cvtColor(
        img_lab_filtrada.astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )

    img_rgb = cv2.cvtColor(
        img_bgr_filtrada,
        cv2.COLOR_BGR2RGB
    )

    return img_rgb


def filtro_minimo(img_bgr, tamanho_kernel=3):
    img_lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float64)

    l, a, b = cv2.split(img_lab)

    h, w = l.shape

    l_filtrado = np.zeros_like(l)

    margem = tamanho_kernel // 2

    l_pad = np.pad(
        l,
        ((margem, margem),
         (margem, margem)),
        mode='edge'
    )

    for i in range(h):
        for j in range(w):
            janela = l_pad[
                i:i + tamanho_kernel,
                j:j + tamanho_kernel
            ]

            l_filtrado[i, j] = np.min(janela)

    img_lab_filtrada = cv2.merge([
        l_filtrado,
        a,
        b
    ])

    img_lab_filtrada = np.clip(
        img_lab_filtrada,
        0,
        255
    ).astype(np.uint8)

    img_bgr_filtrada = cv2.cvtColor(
        img_lab_filtrada,
        cv2.COLOR_LAB2BGR
    )

    img_rgb = cv2.cvtColor(
        img_bgr_filtrada,
        cv2.COLOR_BGR2RGB
    )

    return img_rgb


def filtro_maximo(img_bgr, tamanho_kernel=3):
    img_lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float64)

    l, a, b = cv2.split(img_lab)

    h, w = l.shape

    l_filtrado = np.zeros_like(l)

    margem = tamanho_kernel // 2

    l_pad = np.pad(
        l,
        ((margem, margem),
         (margem, margem)),
        mode='edge'
    )

    for i in range(h):
        for j in range(w):
            janela = l_pad[
                i:i + tamanho_kernel,
                j:j + tamanho_kernel
            ]

            l_filtrado[i, j] = np.max(janela)

    img_lab_filtrada = cv2.merge([
        l_filtrado,
        a,
        b
    ])

    img_lab_filtrada = np.clip(
        img_lab_filtrada,
        0,
        255
    ).astype(np.uint8)

    img_bgr_filtrada = cv2.cvtColor(
        img_lab_filtrada,
        cv2.COLOR_LAB2BGR
    )

    img_rgb = cv2.cvtColor(
        img_bgr_filtrada,
        cv2.COLOR_BGR2RGB
    )

    return img_rgb


def filtro_midpoint(img_bgr, tamanho_kernel=3):
    img_lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float64)

    l, a, b = cv2.split(img_lab)

    h, w = l.shape

    l_filtrado = np.zeros_like(l)

    margem = tamanho_kernel // 2

    l_pad = np.pad(
        l,
        ((margem, margem),
         (margem, margem)),
        mode='edge'
    )

    for i in range(h):
        for j in range(w):
            janela = l_pad[
                i:i + tamanho_kernel,
                j:j + tamanho_kernel
            ]

            l_filtrado[i, j] = (1 / 2) * (np.max(janela) + np.min(janela))

    img_lab_filtrada = cv2.merge([
        l_filtrado,
        a,
        b
    ])

    img_lab_filtrada = np.clip(
        img_lab_filtrada,
        0,
        255
    ).astype(np.uint8)

    img_bgr_filtrada = cv2.cvtColor(
        img_lab_filtrada,
        cv2.COLOR_LAB2BGR
    )

    img_rgb = cv2.cvtColor(
        img_bgr_filtrada,
        cv2.COLOR_BGR2RGB
    )

    return img_rgb


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
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
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
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
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
            pixel_novo   = 255.0 if pixel_antigo >= 127.0 else 0.0
            erro         = pixel_antigo - pixel_novo

            out[i, j] = pixel_novo

            if j + 1 < w:
                out[i,     j + 1] += erro * (7 / 16)
            if i + 1 < h and j - 1 >= 0:
                out[i + 1, j - 1] += erro * (3 / 16)
            if i + 1 < h:
                out[i + 1, j    ] += erro * (5 / 16)
            if i + 1 < h and j + 1 < w:
                out[i + 1, j + 1] += erro * (1 / 16)

    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(out_uint8, cv2.COLOR_GRAY2RGB)

    return img_rgb