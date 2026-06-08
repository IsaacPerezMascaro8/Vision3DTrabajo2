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
                       box_size=BOX_SIZE, escala=0.5, umbral_mult=2.5, morph_k=5, clip_min=0.0, clip_max=128.0):
    """Calcula el mapa de disparidad densa mediante Plane Sweeping."""
    
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

    # --- Pre-procesamiento de Textura (CLAHE) ---
    config.info("[PLANE SWEEP] Aplicando CLAHE para resaltar objetos oscuros...")
    gray1_uint8 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2_uint8 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray1 = clahe.apply(gray1_uint8).astype(np.float32)
    gray2 = clahe.apply(gray2_uint8).astype(np.float32)
    h, w = gray1.shape

    # --- Definir planos de disparidad candidatos ---
    d_candidates = np.linspace(0, max_disp, num_planos)
    volumen_coste = np.full((h, w, num_planos), np.inf, dtype=np.float32)

    for i, d in enumerate(d_candidates):
        d_int = int(round(d))
        if d_int <= 0:
            diff = np.abs(gray1 - gray2)
        else:
            img2_shifted = np.zeros_like(gray2)
            if d_int < w:
                img2_shifted[:, :w - d_int] = gray2[:, d_int:]
                # BORDER REPLICATE
                img2_shifted[:, w - d_int:] = np.tile(gray2[:, -1:], (1, d_int))
            
            diff = np.abs(gray1 - img2_shifted)

        coste_suavizado = cv2.boxFilter(diff, ddepth=-1, ksize=(box_size, box_size), normalize=True)
        volumen_coste[:, :, i] = coste_suavizado

    # --- Selección Discreta y Filtro Guiado ---
    config.info("[PLANE SWEEP] Seleccionando disparidad y aplicando Filtro Guiado...")
    idx_min = np.argmin(volumen_coste, axis=2)
    paso_disp = max_disp / (num_planos - 1)
    disparidad_raw = (idx_min * paso_disp).astype(np.float32)

    disparidad_smooth = guided_filter_disparity(gray1, disparidad_raw, r=25, eps=0.01)

    # --- Máscara de Confianza Estricta ---
    min_cost = np.min(volumen_coste, axis=2)
    umbral = np.mean(min_cost) * umbral_mult

    mask_invalido = ((min_cost > umbral) | (min_cost >= 250)).astype(np.uint8)
    kernel = np.ones((morph_k, morph_k), np.uint8)
    mask_invalido_limpia = cv2.morphologyEx(mask_invalido, cv2.MORPH_OPEN, kernel)

    disparidad_smooth[mask_invalido_limpia == 1] = 0.0
    disparidad_smooth[disparidad_smooth <= 0] = 0.0

    n_validos = int(np.sum(disparidad_smooth > 0))
    config.info(f"[PLANE SWEEP] Píxeles válidos: {n_validos}/{h * w} ({100 * n_validos / (h * w):.1f}%)")

    # --- Recorte de Contraste y Normalización Visual ---
    mask_validos = disparidad_smooth > 0
    disparidad_smooth[mask_validos] = np.clip(disparidad_smooth[mask_validos], clip_min, clip_max)

    disp_norm = cv2.normalize(disparidad_smooth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)
    mapa_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_MAGMA)

    mask_negro = disparidad_smooth <= 0
    mapa_color[mask_negro] = [0, 0, 0]

    # --- Enmascarado de Bordes (Blackout) Definitivo ---
    config.info("[PLANE SWEEP] Apagando zonas muertas para la proyección 3D...")
    margen_izq = int(max_disp)
    margen_der = int(max_disp * 0.25)
    
    if w > (margen_izq + margen_der):
        disparidad_smooth[:, :margen_izq] = 0.0
        disparidad_smooth[:, w - margen_der:] = 0.0
        
        mapa_color[:, :margen_izq] = [0, 0, 0]
        mapa_color[:, w - margen_der:] = [0, 0, 0]

    return disparidad_smooth, mapa_color
