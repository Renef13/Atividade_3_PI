import cv2
import numpy as np

#aplicar o stretching no HSV
def stretching_contraste_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    eps = np.finfo(np.float32).eps

    V = hsv[:, :, 2]
    V = (V - V.min()) * (255 / (V.max() - V.min() + eps))
    V = np.clip(V, 0, 255)

    hsv[:, :, 2] = V

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

#Fazendo o mesmo mas em RGB

def stretching_contraste_rgb(img_rgb):
    img = img_rgb.astype(np.float32)
    eps = np.finfo(np.float32).eps

    out = np.zeros_like(img)

    for c in range(3):
        channel = img[:, :, c]
        out[:, :, c] = (channel - channel.min()) * (255 / (channel.max() - channel.min() + eps))

    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)

def clip_contraste(imagem, p_min=2, p_max=98):
    imagem = imagem.astype(np.float32)
    canais = []

    for c in range(imagem.shape[2]):
        canal = imagem[:, :, c].copy()

        pmin = np.percentile(canal, p_min)
        pmax = np.percentile(canal, p_max)

        np.clip(canal, pmin, pmax, out=canal)

        canal = (canal - pmin) * (255 / (pmax - pmin + np.finfo(np.float32).eps))

        np.clip(canal, 0, 255, out=canal)

        canais.append(canal.astype(np.uint8))

    return np.dstack(canais)

# sem for e sem vetor
# def clip_contraste(imagem, p_min=2, p_max=98):
#     imagem = imagem.astype(np.float32)
#
#     pmin = np.percentile(imagem, p_min, axis=(0, 1), keepdims=True)
#     pmax = np.percentile(imagem, p_max, axis=(0, 1), keepdims=True)
#
#     imagem = np.clip(imagem, pmin, pmax)
#
#     imagem = (imagem - pmin) * (255 / (pmax - pmin + np.finfo(np.float32).eps))
#
#     imagem = np.clip(imagem, 0, 255)
#
#     return imagem.astype(np.uint8)

# funcao de filtro da media
def filtro_media(img, tamanho_kernel=3):
    img = img.astype(np.float64)
    h, w, c = img.shape
    out = np.zeros_like(img)
    margem = tamanho_kernel // 2

    img_pad = np.pad(img, ((margem, margem), (margem, margem), (0, 0)), mode='edge')

    for i in range(h):
        for j in range(w):
            janela = img_pad[i:i + tamanho_kernel, j:j + tamanho_kernel, :]
            media = janela.mean(axis=(0, 1))
            out[i, j] = media

    return out.astype(np.uint8)

# funcao de filtro da mediana
def filtro_mediana(img, tamanho_kernel=3):
    img = img.astype(np.float64)
    h, w, c = img.shape
    out = np.zeros_like(img)
    margem = tamanho_kernel // 2
    img_pad = np.pad(img, ((margem, margem), (margem, margem), (0, 0)), mode='edge')

    for i in range(h):
        for j in range(w):
            janela = img_pad[i:i + tamanho_kernel, j:j + tamanho_kernel, :]
            mediana = np.median(janela, axis=(0, 1))
            out[i, j] = mediana

    return np.clip(out, 0, 255, ).astype(np.uint8)

#função de especificação de histograma
def esp_hist(referencia, alvo):

    referencia_lab = cv2.cvtColor(referencia, cv2.COLOR_BGR2LAB)
    alvo_lab = cv2.cvtColor(alvo, cv2.COLOR_BGR2LAB)

    l_ref, a_ref, b_ref = cv2.split(referencia_lab)
    l_alvo, a_alvo, b_alvo = cv2.split(alvo_lab)

    hist_ref = cv2.calcHist([l_ref], [0],None, [256], [0, 256])
    hist_alvo = cv2.calcHist([l_alvo], [0],None, [256], [0, 256])

    pdf_ref = hist_ref / hist_ref.sum()
    pdf_alvo = hist_alvo / hist_alvo.sum()

    cdf_ref = np.cumsum(pdf_ref)
    cdf_alvo = np.cumsum(pdf_alvo)

    mapeamento = np.interp(cdf_alvo, cdf_ref,range(256))

    l_novo = cv2.LUT(l_alvo,mapeamento.astype(np.uint8))

    alvo_novo_lab = cv2.merge([l_novo, a_alvo,b_alvo])

    imagem_resultado = cv2.cvtColor(alvo_novo_lab, cv2.COLOR_LAB2BGR)

    return imagem_resultado

