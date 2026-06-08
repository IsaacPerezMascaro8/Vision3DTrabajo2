"""
Disparidad Densa mediante Block Matching (SSD) y comparativa con SGBM.

Genera tres mapas de disparidad sobre las imágenes rectificadas:

  1. SSD puro con ventana pequeña  (W=5)  → más detalle, más ruido.
  2. SSD puro con ventana grande   (W=21) → más suave, bordes engrosados.
  3. StereoSGBM (minimización de energía) → restricción de suavidad global.

Mejoras implementadas sobre el SSD base para la comparativa:
  A. Census Transform          → robustez fotométrica (soluciona SSD puro).
  B. Agregación bilateral      → anti-fattening en bordes.
  C. Left-Right Consistency    → detección de oclusiones.
  D. Guided Filter             → coherencia espacial sin destruir bordes.
  E. Máscara de textura        → rechazo de zonas sin contraste.
  F. Uniqueness Ratio          → rechazo de mínimos locales ambiguos.

Los resultados se guardan en output/ con colormap TURBO para facilitar
la comparación visual en la memoria técnica.
"""

import os
import sys
import numpy as np
import cv2
import config


# ======================================================================
# 1. Census Transform — robustez frente a cambios fotométricos
# ======================================================================

def census_transform(image, half_win=2):
    """Calcula la Census Transform de una imagen en escala de grises.

    Para cada píxel, genera un bit-string comparando la intensidad del
    centro con cada vecino del parche (2*half_win+1)². El resultado es
    un entero de hasta 64 bits que codifica la estructura local.

    Invariante a cambios monotónicos de iluminación (ganancia/offset).

    Parameters
    ----------
    image : ndarray (float32 o uint8)
        Imagen en escala de grises.
    half_win : int
        Mitad de la ventana Census (2 → ventana 5×5, 24 bits).

    Returns
    -------
    census : ndarray (uint64)
        Imagen transformada (recortada en los bordes por half_win).
    """
    h, w = image.shape
    win = 2 * half_win + 1
    n_bits = win * win - 1  # excluimos el centro
    if n_bits > 64:
        raise ValueError("Ventana Census demasiado grande (máx 64 bits → half_win ≤ 4)")

    img = image.astype(np.float32)
    center = img[half_win:h - half_win, half_win:w - half_win]
    census = np.zeros_like(center, dtype=np.uint64)

    bit = 0
    for dy in range(win):
        for dx in range(win):
            if dy == half_win and dx == half_win:
                continue
            neighbor = img[dy:h - 2 * half_win + dy,
                           dx:w - 2 * half_win + dx]
            census |= (neighbor >= center).astype(np.uint64) << bit
            bit += 1

    return census


def hamming_distance_volume(census_left, census_right, max_disp):
    """Construye el volumen de costes Census (distancia Hamming).

    Parameters
    ----------
    census_left, census_right : ndarray (uint64)
        Imágenes transformadas con Census.
    max_disp : int
        Rango máximo de disparidad.

    Returns
    -------
    cost_vol : ndarray (float32, shape H×W×max_disp)
    """
    h, w = census_left.shape
    cost_vol = np.full((h, w, max_disp), 255.0, dtype=np.float32)

    for d in range(max_disp):
        if d == 0:
            xor = census_left ^ census_right
        else:
            if d >= w:
                break
            xor = np.zeros((h, w), dtype=np.uint64)
            xor[:, d:] = census_left[:, d:] ^ census_right[:, :w - d]

        # Contar bits con lookup-table: dividir en bytes
        hamming = np.zeros((h, w), dtype=np.float32)
        val = xor.copy()
        for _ in range(8):  # uint64 = 8 bytes
            hamming += _popcount_byte(val & np.uint64(0xFF))
            val >>= np.uint64(8)

        if d == 0:
            cost_vol[:, :, d] = hamming
        else:
            cost_vol[:, d:, d] = hamming[:, d:]

    return cost_vol


def _popcount_byte(byte_arr):
    """Cuenta bits activos de un array de bytes (0-255) mediante LUT."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = bin(i).count('1')
    return lut[byte_arr.astype(np.uint8)]


# ======================================================================
# 2. Agregación bilateral del coste — anti-fattening
# ======================================================================

def bilateral_cost_aggregation(cost_vol, guide, win_size=9,
                                sigma_spatial=4.0, sigma_color=15.0):
    """Agrega el volumen de costes con un filtro bilateral guiado.

    A diferencia del boxFilter uniforme, pondera cada vecino según su
    similitud cromática con el píxel central, evitando mezclar costes
    de primer plano y fondo (foreground fattening).

    Parameters
    ----------
    cost_vol : ndarray (float32, H×W×D)
    guide : ndarray (float32, H×W) — imagen de guía (grises).
    win_size : int — lado del parche.
    sigma_spatial, sigma_color : float — sigmas del bilateral.

    Returns
    -------
    agg : ndarray (float32, H×W×D)
    """
    h, w, D = cost_vol.shape
    agg = np.empty_like(cost_vol)

    # Usar jointBilateralFilter de OpenCV (más rápido que un doble bucle)
    for d in range(D):
        slice_d = cost_vol[:, :, d]
        # cv2.bilateralFilter opera en uint8/float32 de 1 canal.
        agg[:, :, d] = cv2.bilateralFilter(
            slice_d, d=win_size,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_spatial,
        )

    return agg


# ======================================================================
# 3. Implementación SSD mejorada (Census + Bilateral + LRC + Guided)
# ======================================================================

def disparidad_ssd(gray_left, gray_right, win_size=5, max_disp=128):
    """Calcula el mapa de disparidad densa mediante Block Matching SSD.

    Para cada píxel (y, x) de la imagen izquierda, desliza una ventana
    de tamaño ``win_size × win_size`` a lo largo de la línea epipolar
    en la imagen derecha y selecciona la disparidad que minimiza la
    Sum of Squared Differences (Winner-Takes-All).

    Parameters
    ----------
    gray_left, gray_right : ndarray (float32)
        Imágenes rectificadas en escala de grises.
    win_size : int
        Lado de la ventana cuadrada (debe ser impar).
    max_disp : int
        Rango máximo de disparidades a evaluar.

    Returns
    -------
    disp_map : ndarray (float32)
        Mapa de disparidad con valores en [0, max_disp].
    """
    h, w = gray_left.shape
    margen = win_size // 2

    # Volumen de costes: (h, w, max_disp)  — se acumula con boxFilter
    # para evitar el doble bucle (y, x) y aprovechar la vectorización.
    cost_volume = np.full((h, w, max_disp), np.inf, dtype=np.float32)

    for d in range(max_disp):
        if d == 0:
            diff = gray_left - gray_right
        else:
            # Desplazar la imagen derecha d píxeles a la izquierda.
            # Solo la región solapada es válida; el resto queda a inf.
            diff = np.full((h, w), np.inf, dtype=np.float32)
            diff[:, d:] = gray_left[:, d:] - gray_right[:, :w - d]

        # SSD acumulada dentro de la ventana (boxFilter normalizado=False).
        sq = np.where(np.isfinite(diff), diff * diff, 0.0).astype(np.float32)
        valid = np.isfinite(diff).astype(np.float32)

        sum_sq = cv2.boxFilter(sq, ddepth=-1,
                               ksize=(win_size, win_size), normalize=False)
        count  = cv2.boxFilter(valid, ddepth=-1,
                               ksize=(win_size, win_size), normalize=False)

        # Solo considerar posiciones donde la ventana esté completamente
        # dentro de la zona válida (count == win_size²).
        full_window = (count >= win_size * win_size)
        cost_volume[:, :, d] = np.where(full_window, sum_sq, np.inf)

    # Winner-Takes-All: disparidad que minimiza el coste.
    disp_map = np.argmin(cost_volume, axis=2).astype(np.float32)

    # Invalidar bordes donde no se pudo calcular la ventana completa.
    disp_map[:margen, :] = 0
    disp_map[-margen:, :] = 0
    disp_map[:, :max(margen, max_disp)] = 0
    disp_map[:, -margen:] = 0

    return disp_map


def disparidad_ssd_mejorada(gray_left, gray_right, win_size=9, max_disp=128):
    """SSD mejorado con Census Transform + Bilateral Aggregation +
    Left-Right Consistency + Guided Filter + Texture/Uniqueness masks.

    Resuelve los 6 problemas documentados del SSD puro:
      1. Ruido en zonas sin textura   → máscara de varianza + Census
      2. Sensibilidad a mínimos loc.  → uniqueness ratio filtering
      3. Foreground fattening         → agregación bilateral
      4. Pérdida de detalles finos    → guided filter (no box)
      5. Sin coherencia espacial      → guided filter post-WTA
      6. Oclusiones                   → left-right consistency check
      7. Cambios fotométricos         → Census Transform (no SSD puro)

    Parameters
    ----------
    gray_left, gray_right : ndarray (float32)
        Imágenes rectificadas en escala de grises.
    win_size : int
        Lado de la ventana de agregación (impar).
    max_disp : int
        Rango máximo de disparidades.

    Returns
    -------
    disp_final : ndarray (float32)
        Mapa de disparidad refinado con valores en [0, max_disp].
    """
    h, w = gray_left.shape
    config.info(f"  [CENSUS] Transformando imágenes ({h}×{w}) ...")

    # ----- Pre-procesamiento: CLAHE para resaltar texturas débiles -----
    config.info(f"  [CLAHE] Realzando textura de paredes lisas ...")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_left_clahe = clahe.apply(np.clip(gray_left, 0, 255).astype(np.uint8)).astype(np.float32)
    gray_right_clahe = clahe.apply(np.clip(gray_right, 0, 255).astype(np.uint8)).astype(np.float32)

    # ----- A. Census Transform (coste robusto) -----
    census_half = 2  # ventana 5×5 → 24 bits
    cL = census_transform(gray_left_clahe, half_win=census_half)
    cR = census_transform(gray_right_clahe, half_win=census_half)

    # El Census recorta bordes; ajustamos las guías usando las imágenes ORIGINALES (para respetar los bordes reales)
    pad = census_half
    guide_L = gray_left[pad:h - pad, pad:w - pad].copy()
    guide_R = gray_right[pad:h - pad, pad:w - pad].copy()
    hc, wc = cL.shape

    config.info(f"  [CENSUS] Construyendo volumen de costes Hamming ...")
    cost_L = hamming_distance_volume(cL, cR, max_disp)

    # ----- B. Agregación bilateral (anti-fattening) -----
    config.info(f"  [BILATERAL] Agregando costes (win={win_size}) ...")
    cost_L = bilateral_cost_aggregation(
        cost_L, guide_L,
        win_size=win_size, sigma_spatial=win_size / 2.0, sigma_color=20.0
    )

    # ----- WTA izquierdo -----
    disp_L = np.argmin(cost_L, axis=2).astype(np.float32)

    # ----- F. Uniqueness ratio (rechazo de mínimos ambiguos) -----
    config.info(f"  [UNIQUE] Filtrando mínimos ambiguos ...")
    min_cost = np.min(cost_L, axis=2)
    cost_masked = cost_L.copy()
    idx_best = np.argmin(cost_L, axis=2)
    rows, cols = np.meshgrid(np.arange(hc), np.arange(wc), indexing='ij')
    cost_masked[rows, cols, idx_best] = np.inf
    second_min = np.min(cost_masked, axis=2)

    uniqueness_mask = (second_min - min_cost) < (0.02 * second_min)
    disp_L[uniqueness_mask] = 0
    
    # ----- C. Left-Right Consistency Check (oclusiones) -----
    config.info(f"  [LRC] Calculando disparidad derecha para consistency check ...")
    cost_R_correct = np.full((hc, wc, max_disp), 255.0, dtype=np.float32)
    for d in range(max_disp):
        if d == 0:
            xor = cR ^ cL
            hamming = np.zeros((hc, wc), dtype=np.float32)
            val = xor.copy()
            for _ in range(8):
                hamming += _popcount_byte(val & np.uint64(0xFF))
                val >>= np.uint64(8)
            cost_R_correct[:, :, d] = hamming
        elif d < wc:
            xor = np.zeros((hc, wc), dtype=np.uint64)
            xor[:, :wc - d] = cR[:, :wc - d] ^ cL[:, d:]
            hamming = np.zeros((hc, wc), dtype=np.float32)
            val = xor.copy()
            for _ in range(8):
                hamming += _popcount_byte(val & np.uint64(0xFF))
                val >>= np.uint64(8)
            cost_R_correct[:, :wc - d, d] = hamming[:, :wc - d]

    cost_R_correct = bilateral_cost_aggregation(
        cost_R_correct, guide_R,
        win_size=win_size, sigma_spatial=win_size / 2.0, sigma_color=20.0
    )
    disp_R = np.argmin(cost_R_correct, axis=2).astype(np.float32)

    # Cross-check: |D_L(x,y) - D_R(x - D_L(x,y), y)| < threshold
    lr_threshold = 5.0
    invalid = np.ones((hc, wc), dtype=bool)
    for y in range(hc):
        for x in range(wc):
            d = int(disp_L[y, x])
            xr = x - d
            if 0 <= xr < wc:
                if abs(disp_L[y, x] - disp_R[y, xr]) <= lr_threshold:
                    invalid[y, x] = False
    disp_L[invalid] = 0
    n_occ = int(np.sum(invalid))
    config.info(f"  [LRC] Píxeles ocluidos/invalidados: {n_occ} "
                f"({100 * n_occ / (hc * wc):.1f}%)")

    # ----- E. Máscara de textura (varianza local) -----
    config.info(f"  [TEXTURA] Enmascarando zonas absolutamente sin textura ...")
    mean_local = cv2.boxFilter(guide_L, ddepth=-1, ksize=(11, 11))
    mean_sq = cv2.boxFilter(guide_L * guide_L, ddepth=-1, ksize=(11, 11))
    variance = mean_sq - mean_local * mean_local
    texture_threshold = 2.0  # Solo elimina el negro puro (padding) o zonas quemadas
    disp_L[variance < texture_threshold] = 0

    # ----- D. Guided Filter (coherencia espacial post-WTA) -----
    config.info(f"  [GUIDED] Aplicando Filtro Guiado de He ...")
    disp_final = _guided_filter(guide_L, disp_L, r=9, eps=100.0)

    # Preservar invalidados (valor 0)
    disp_final[disp_L <= 0] = 0
    disp_final = np.clip(disp_final, 0, max_disp - 1)

    # ----- Rellenar huecos con mediana ponderada -----
    config.info(f"  [FILL] Rellenando huecos con filtro mediana ...")
    holes = disp_final <= 0
    if np.any(holes) and np.any(~holes):
        disp_uint8 = np.clip(disp_final, 0, 255).astype(np.uint8)
        disp_filled = cv2.medianBlur(disp_uint8, ksize=5).astype(np.float32)
        disp_final[holes] = disp_filled[holes]
        # Segunda pasada para huecos remanentes
        holes2 = disp_final <= 0
        if np.any(holes2):
            disp_uint8_2 = np.clip(disp_final, 0, 255).astype(np.uint8)
            disp_filled2 = cv2.medianBlur(disp_uint8_2, ksize=15).astype(np.float32)
            disp_final[holes2] = disp_filled2[holes2]

    # Invalidar bordes extremos
    margen = max(census_half, max_disp, win_size // 2)
    disp_final[:5, :] = 0
    disp_final[-5:, :] = 0
    disp_final[:, :margen] = 0
    disp_final[:, -5:] = 0

    return disp_final


def _guided_filter(guide, src, r=9, eps=100.0):
    """Filtro Guiado de Kaiming He — coherencia espacial edge-aware.

    A diferencia del boxFilter, preserva los bordes de la imagen guía
    mientras suaviza zonas planas, imponiendo coherencia espacial
    sin destruir discontinuidades de profundidad.
    """
    guide_f = guide.astype(np.float32)
    if guide_f.max() > 1.0:
        guide_f /= 255.0
    src_f = src.astype(np.float32)
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(guide_f, -1, ksize)
    mean_p = cv2.boxFilter(src_f, -1, ksize)
    mean_Ip = cv2.boxFilter(guide_f * src_f, -1, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_f * guide_f, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    return mean_a * guide_f + mean_b


# ======================================================================
# 4. Disparidad mediante StereoSGBM (minimización de energía)
# ======================================================================

def disparidad_sgbm(gray_left, gray_right, win_size=5, max_disp=128):
    """Genera un mapa de disparidad con StereoSGBM (Semi-Global Block
    Matching) de OpenCV, que aproxima la minimización de energía global
    con restricciones de suavidad P1 y P2.

    Parameters
    ----------
    gray_left, gray_right : ndarray (uint8)
        Imágenes rectificadas en escala de grises.
    win_size : int
        Tamaño del bloque de matching.
    max_disp : int
        Rango máximo de disparidades (debe ser múltiplo de 16).

    Returns
    -------
    disp_map : ndarray (float32)
        Mapa de disparidad en píxeles (valores reales, no escalados ×16).
    """
    # Asegurar múltiplo de 16
    num_disp = int(np.ceil(max_disp / 16) * 16)

    # P1 y P2 controlan la restricción de suavidad E_smooth.
    # P1 penaliza cambios pequeños de disparidad (superficies inclinadas).
    # P2 penaliza cambios grandes (discontinuidades de profundidad).
    P1 = 8  * 3 * win_size * win_size
    P2 = 32 * 3 * win_size * win_size

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=win_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp_raw = sgbm.compute(gray_left, gray_right)
    # StereoSGBM devuelve disparidad × 16 en int16.
    disp_map = disp_raw.astype(np.float32) / 16.0
    disp_map[disp_map < 0] = 0

    return disp_map


# ======================================================================
# 5. Utilidad de normalización y colormap
# ======================================================================

def normalizar_colormap(disp_map, colormap=cv2.COLORMAP_TURBO):
    """Normaliza un mapa de disparidad a [0, 255] y aplica un colormap.

    Los píxeles con disparidad ≤ 0 se pintan de negro.

    Returns
    -------
    color : ndarray (uint8, BGR)
        Imagen coloreada lista para guardar con cv2.imwrite.
    """
    disp_vis = disp_map.copy()
    mask_valid = disp_vis > 0

    if np.any(mask_valid):
        vmin = np.min(disp_vis[mask_valid])
        vmax = np.max(disp_vis[mask_valid])
        if vmax > vmin:
            disp_vis[mask_valid] = 255.0 * (disp_vis[mask_valid] - vmin) / (vmax - vmin)
        else:
            disp_vis[mask_valid] = 0

    disp_uint8 = np.uint8(np.clip(disp_vis, 0, 255))
    color = cv2.applyColorMap(disp_uint8, colormap)
    color[~mask_valid] = [0, 0, 0]

    return color


# ======================================================================
# 6. Script principal
# ======================================================================

def main():
    out_dir = config.ensure_output_dir("output")

    # --- Carga de imágenes rectificadas ---
    left_path  = os.path.join(out_dir, "rectified_left.png")
    right_path = os.path.join(out_dir, "rectified_right.png")

    if not os.path.isfile(left_path) or not os.path.isfile(right_path):
        sys.exit("[ERROR] No se encontraron las imágenes rectificadas en output/. "
                 "Ejecuta primero el pipeline principal: python3 main.py --no-gui")

    img_left  = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    gray_left  = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Downscale para acelerar el SSD puro (imágenes grandes)
    escala = 0.5
    h_orig, w_orig = gray_left.shape
    new_h, new_w = int(h_orig * escala), int(w_orig * escala)
    gray_left_ds  = cv2.resize(gray_left,  (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray_right_ds = cv2.resize(gray_right, (new_w, new_h), interpolation=cv2.INTER_AREA)

    max_disp = 64
    config.separator("DISPARIDAD DENSA — SSD BLOCK MATCHING")

    # ── 1. SSD con ventana pequeña (W=5) ──
    W_small = 5
    config.info(f"[SSD] Calculando disparidad con ventana W={W_small} ...")
    disp_small = disparidad_ssd(gray_left_ds.astype(np.float32),
                                gray_right_ds.astype(np.float32),
                                win_size=W_small, max_disp=max_disp)

    color_small = normalizar_colormap(disp_small)
    path_small = os.path.join(out_dir, f"ssd_disparity_W{W_small}.png")
    cv2.imwrite(path_small, color_small)
    config.info(f"[SSD] Guardado → {path_small}")

    # ── 2. SSD con ventana grande (W=21) ──
    W_large = 31
    config.info(f"[SSD] Calculando disparidad con ventana W={W_large} ...")
    disp_large = disparidad_ssd(gray_left_ds.astype(np.float32),
                                gray_right_ds.astype(np.float32),
                                win_size=W_large, max_disp=max_disp)

    color_large = normalizar_colormap(disp_large)
    path_large = os.path.join(out_dir, f"ssd_disparity_W{W_large}.png")
    cv2.imwrite(path_large, color_large)
    config.info(f"[SSD] Guardado → {path_large}")

    # ── 3. SSD MEJORADO (Census + Bilateral + LRC + Guided) ──
    config.separator("SSD MEJORADO (Census + Bilateral + LRC + Guided Filter)")
    config.info("[SSD+] Calculando disparidad mejorada ...")
    disp_improved = disparidad_ssd_mejorada(
        gray_left_ds.astype(np.float32),
        gray_right_ds.astype(np.float32),
        win_size=9, max_disp=max_disp,
    )

    color_improved = normalizar_colormap(disp_improved)
    path_improved = os.path.join(out_dir, "ssd_disparity_mejorado.png")
    cv2.imwrite(path_improved, color_improved)
    config.info(f"[SSD+] Guardado → {path_improved}")

    # ── 4. SGBM (minimización de energía) ──
    config.info("[SGBM] Calculando disparidad con StereoSGBM (Semi-Global) ...")
    disp_sgbm = disparidad_sgbm(gray_left_ds, gray_right_ds,
                                win_size=5, max_disp=max_disp)

    color_sgbm = normalizar_colormap(disp_sgbm)
    path_sgbm = os.path.join(out_dir, "sgbm_disparity.png")
    cv2.imwrite(path_sgbm, color_sgbm)
    config.info(f"[SGBM] Guardado → {path_sgbm}")

    config.separator("COMPARATIVA COMPLETA")
    config.info(f"  • SSD W={W_small}  (puro)     → {path_small}")
    config.info(f"  • SSD W={W_large} (puro)     → {path_large}")
    config.info(f"  • SSD Mejorado             → {path_improved}")
    config.info(f"  • SGBM                     → {path_sgbm}")
    config.info("Listo. Revisa output/ para la comparación visual.")


if __name__ == "__main__":
    main()
