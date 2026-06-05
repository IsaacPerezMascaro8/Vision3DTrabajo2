"""
Plane Sweeping Estéreo para correspondencia densa.

Este algoritmo no depende de textura local como SSD/SGBM, sino de la
coherencia de profundidad: proyecta la imagen derecha sobre un conjunto
de planos de disparidad candidatos, acumula el error SAD con un filtro
de suavizado y selecciona el plano ganador por píxel.

Ideal para escenas con superficies planas, reflectantes o con poca textura.
"""
import numpy as np
import cv2
import config

NUM_PLANOS = 96
MAX_DISP = 128
BOX_SIZE = 7


def guided_filter_disparity(I, p, r=25, eps=1e-3):
    """
    Filtro Guiado de Kaiming He.
    I: Imagen de guía (escala de grises original, normalizada)
    p: Mapa de disparidad con escaleras (float32)
    """
    I_float = I.astype(np.float32) / 255.0

    mean_I = cv2.boxFilter(I_float, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I_float * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I_float * I_float, -1, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))

    q = mean_a * I_float + mean_b
    return q


def plane_sweep_stereo(img_rect1, img_rect2, num_planos=NUM_PLANOS, max_disp=MAX_DISP,
                       box_size=BOX_SIZE, escala=0.5):
    """Calcula el mapa de disparidad densa mediante Plane Sweeping.

    Parameters
    ----------
    img_rect1, img_rect2 : ndarray (BGR, uint8)
        Par de imágenes rectificadas.
    num_planos : int
        Número de planos de profundidad candidatos a evaluar.
    max_disp : int
        Disparidad máxima en píxeles (tras el downsampling).
    box_size : int
        Tamaño del filtro de caja para suavizar el volumen de coste.
    escala : float
        Factor de downsampling para reducir el coste computacional.

    Returns
    -------
    disparidad_final : ndarray (float32)
        Mapa de disparidad en píxeles (escala reducida).
    mapa_color : ndarray (uint8, 3 canales)
        Versión coloreada con COLORMAP_MAGMA.
    """
    # --- Downsampling para eficiencia ---
    if escala != 1.0:
        h_orig, w_orig = img_rect1.shape[:2]
        new_h, new_w = int(h_orig * escala), int(w_orig * escala)
        img1 = cv2.resize(img_rect1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img_rect2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        config.info(f"[PLANE SWEEP] Downsampling: {w_orig}x{h_orig} → {new_w}x{new_h}")
    else:
        img1 = img_rect1
        img2 = img_rect2

    # Convertir a escala de grises float32
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray1.shape

    # --- Definir planos de disparidad candidatos ---
    d_candidates = np.linspace(0, max_disp, num_planos)
    config.info(f"[PLANE SWEEP] {num_planos} planos, disparidad 0–{max_disp} px, box={box_size}")

    # --- Construir volumen de coste (H, W, num_planos) ---
    volumen_coste = np.full((h, w, num_planos), np.inf, dtype=np.float32)

    for i, d in enumerate(d_candidates):
        d_int = int(round(d))
        if d_int <= 0:
            # Disparidad 0 → imágenes alineadas
            diff = np.abs(gray1 - gray2)
        else:
            # Desplazar img2 horizontalmente d píxeles hacia la izquierda
            img2_shifted = np.zeros_like(gray2)
            if d_int < w:
                img2_shifted[:, :w - d_int] = gray2[:, d_int:]
            diff = np.abs(gray1 - img2_shifted)
            # Marcar la zona sin datos (borde derecho) como coste alto
            diff[:, w - d_int:] = 255.0

        # Suavizar el error con un filtro de caja (SAD sobre una región)
        coste_suavizado = cv2.boxFilter(diff, ddepth=-1,
                                        ksize=(box_size, box_size),
                                        normalize=True)
        volumen_coste[:, :, i] = coste_suavizado

        # Progreso
        if (i + 1) % max(1, num_planos // 5) == 0:
            config.info(f"[PLANE SWEEP]   Plano {i + 1}/{num_planos} (d={d:.1f} px)")

    # --- Selección Discreta y Filtro Guiado ---
    config.info("[PLANE SWEEP] Seleccionando disparidad y aplicando Filtro Guiado...")
    idx_min = np.argmin(volumen_coste, axis=2)

    paso_disp = max_disp / (num_planos - 1)
    disparidad_raw = (idx_min * paso_disp).astype(np.float32)

    # LA MAGIA: Aplicamos el filtro guiado usando la imagen en grises como guía
    disparidad_smooth = guided_filter_disparity(gray1, disparidad_raw, r=25, eps=0.01)

    # --- Máscara de Confianza (Proteger el suelo) ---
    min_cost = np.min(volumen_coste, axis=2)
    umbral = np.mean(min_cost) * 1.1
    mask_invalido = (min_cost > umbral).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask_invalido_limpia = cv2.morphologyEx(mask_invalido, cv2.MORPH_OPEN, kernel)

    disparidad_smooth[mask_invalido_limpia == 1] = 0.0
    disparidad_smooth[disparidad_smooth <= 0] = 0.0

    n_validos = int(np.sum(disparidad_smooth > 0))
    n_total = h * w
    config.info(f"[PLANE SWEEP] Píxeles válidos: {n_validos}/{n_total} "
                f"({100 * n_validos / n_total:.1f}%)")

    # --- Normalización Visual ---
    disp_norm = cv2.normalize(disparidad_smooth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)
    mapa_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_MAGMA)

    mask_negro = disparidad_smooth <= 0
    mapa_color[mask_negro] = [0, 0, 0]

    config.info("[PLANE SWEEP] Plane sweeping completado.")
    return disparidad_smooth, mapa_color
