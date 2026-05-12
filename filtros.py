import numpy as np
import cv2

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

def filtro_gaussiana(img_bgr, tamanho_kernel=3):
