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
