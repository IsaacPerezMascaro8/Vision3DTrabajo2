"""
geometria_epipolar.py
=====================
Módulo de geometría epipolar:
  - Detección de esquinas de marcadores ArUco en un par estéreo.
  - Algoritmo de los 8 puntos normalizado (implementación NumPy pura).
  - Estimación robusta de la Matriz Fundamental (F) con RANSAC.
  - Cálculo de la Matriz Esencial (E) forzando valores singulares.
"""

import numpy as np
import cv2
import config


# ---------------------------------------------------------------------------
# 1. Detección de correspondencias ArUco
# ---------------------------------------------------------------------------

def _corner_sharpness(gray, corners):
    """
    Evalúa la calidad de las esquinas midiendo la magnitud media del gradiente
    alrededor de cada esquina. Cuanto mayor, más nítida y fiable la detección.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    total = 0.0
    for (x, y) in corners.reshape(-1, 2):
        xi, yi = int(round(x)), int(round(y))
        x0 = max(0, xi - 3); x1 = min(gray.shape[1] - 1, xi + 3)
        y0 = max(0, yi - 3); y1 = min(gray.shape[0] - 1, yi + 3)
        patch = mag[y0:y1 + 1, x0:x1 + 1]
        total += float(np.mean(patch)) if patch.size > 0 else 0.0
    return total


def _build_aggressive_params():
    """Construye parámetros agresivos del detector ArUco."""
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 93
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.005
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.005
    params.minDistanceToBorder = 0
    params.minMarkerDistanceRate = 0.005
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    return params


def detectar_aruco(imagen, diccionario=cv2.aruco.DICT_4X4_50, refine_corners=False,
                   max_valid_id=8):
    """
    Detecta marcadores ArUco en una imagen usando detección en TRES PASADAS
    para capturar marcadores difíciles (bajo contraste, mala iluminación).

    1ª pasada: Parámetros estándar sobre la imagen original.
    2ª pasada: Parámetros agresivos sobre la imagen original.
    3ª pasada: Imagen realzada con CLAHE + parámetros agresivos.

    Para marcadores detectados en varias pasadas, se elige la detección
    con mejor calidad de esquinas (mayor gradiente = más nítido).
    Filtro: Se descartan IDs > max_valid_id para evitar falsos positivos.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(diccionario)
    gray_original = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # ===================================================================
    # 1ª PASADA — Detección estándar
    # ===================================================================
    params_std = cv2.aruco.DetectorParameters()
    params_std.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params_std.cornerRefinementWinSize = 5
    params_std.cornerRefinementMaxIterations = 50
    detector_std = cv2.aruco.ArucoDetector(aruco_dict, params_std)
    corners_1, ids_1, _ = detector_std.detectMarkers(imagen)

    pasada1 = {}
    if ids_1 is not None:
        for i, marker_id in enumerate(ids_1.ravel()):
            pasada1[int(marker_id)] = corners_1[i].reshape(4, 2)
    config.info(f"[INFO] 1ª pasada (estándar): IDs {sorted(pasada1.keys())}")

    # ===================================================================
    # 2ª PASADA — Parámetros agresivos sobre imagen ORIGINAL
    # ===================================================================
    params_agr = _build_aggressive_params()
    detector_agr = cv2.aruco.ArucoDetector(aruco_dict, params_agr)
    corners_2, ids_2, _ = detector_agr.detectMarkers(imagen)

    pasada2 = {}
    if ids_2 is not None:
        for i, marker_id in enumerate(ids_2.ravel()):
            pasada2[int(marker_id)] = corners_2[i].reshape(4, 2)
    config.info(f"[INFO] 2ª pasada (agresiva original): IDs {sorted(pasada2.keys())}")

    # ===================================================================
    # 3ª PASADA — CLAHE + parámetros agresivos
    # ===================================================================
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_original)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    params_clahe = _build_aggressive_params()
    detector_clahe = cv2.aruco.ArucoDetector(aruco_dict, params_clahe)
    corners_3, ids_3, _ = detector_clahe.detectMarkers(enhanced_bgr)

    pasada3 = {}
    if ids_3 is not None:
        for i, marker_id in enumerate(ids_3.ravel()):
            pasada3[int(marker_id)] = corners_3[i].reshape(4, 2)
    config.info(f"[INFO] 3ª pasada (CLAHE agresiva): IDs {sorted(pasada3.keys())}")

    # ===================================================================
    # FUSIÓN — Para cada marker, elegir la mejor detección por calidad
    # ===================================================================
    all_ids = set(pasada1.keys()) | set(pasada2.keys()) | set(pasada3.keys())
    esquinas_por_id = {}

    for mid in all_ids:
        candidates = []
        if mid in pasada1:
            candidates.append(('P1', pasada1[mid]))
        if mid in pasada2:
            candidates.append(('P2', pasada2[mid]))
        if mid in pasada3:
            # Re-refinar sobre la imagen original (no CLAHE)
            pts_ref = pasada3[mid].astype(np.float32).reshape(-1, 1, 2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            cv2.cornerSubPix(gray_original, pts_ref, winSize=(7, 7),
                             zeroZone=(-1, -1), criteria=criteria)
            candidates.append(('P3', pts_ref.reshape(4, 2)))

        # Elegir la detección con mayor sharpness (gradiente más fuerte)
        best_name, best_corners = max(candidates,
                                      key=lambda c: _corner_sharpness(gray_original, c[1]))
        esquinas_por_id[mid] = best_corners

        if len(candidates) > 1:
            names = [c[0] for c in candidates]
            config.info(f"[INFO]   Marker {mid}: detectado en {names}, mejor → {best_name}")

    # ===================================================================
    # FILTRO DE IDs — Evitar falsos positivos
    # ===================================================================
    esquinas_por_id = {k: v for k, v in esquinas_por_id.items() if k <= max_valid_id}
    config.info(f"[INFO] ArUco final (tras filtro ID<={max_valid_id}): {sorted(esquinas_por_id.keys())}")

    # ===================================================================
    # Refinamiento subpíxel final sobre imagen original
    # ===================================================================
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    for mid in list(esquinas_por_id.keys()):
        pts = esquinas_por_id[mid].astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray_original, pts, winSize=(7, 7),
                         zeroZone=(-1, -1), criteria=criteria)
        esquinas_por_id[mid] = pts.reshape(4, 2)

    return esquinas_por_id


def alinear_esquinas_cyclic(c1, c2):
    """
    Alinea las 4 esquinas de c2 con respecto a c1 probando desplazamientos
    cíclicos (0..3) y eligiendo el que minimiza la suma de distancias.

    Esta estrategia respeta el ordering devuelto por OpenCV y corrige
    rotaciones (cyclic shifts) del marcador entre las dos imágenes.
    """
    best_shift = 0
    best_cost = float('inf')
    for s in range(4):
        c2s = np.roll(c2, s, axis=0)
        cost = np.sum(np.linalg.norm(c1 - c2s, axis=1))
        if cost < best_cost:
            best_cost = cost
            best_shift = s
    return np.roll(c2, best_shift, axis=0)


def compute_corner_quality(img, corners_dict):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    median_grad = np.median(mag)
    qualities = {}
    for mid, c in corners_dict.items():
        q_per_corner = []
        for (x, y) in c:
            xi = int(round(x)); yi = int(round(y))
            x0 = max(0, xi-2); x1 = min(gray.shape[1]-1, xi+2)
            y0 = max(0, yi-2); y1 = min(gray.shape[0]-1, yi+2)
            patch = mag[y0:y1+1, x0:x1+1]
            if patch.size == 0:
                q = 0.0
            else:
                q = float(np.mean(patch))
            if median_grad > 1e-6:
                q = q / (median_grad)
            q_per_corner.append(q)
        qualities[mid] = q_per_corner
    return qualities

def obtener_correspondencias_aruco(img1, img2,
                                   diccionario=cv2.aruco.DICT_4X4_50,
                                   refine_corners=False,
                                   min_corner_quality=0.0):
    """
    Encuentra correspondencias de puntos entre dos imágenes usando
    las esquinas de los marcadores ArUco comunes.
    """
    esquinas1 = detectar_aruco(img1, diccionario, refine_corners=refine_corners)
    esquinas2 = detectar_aruco(img2, diccionario, refine_corners=refine_corners)

    if min_corner_quality and min_corner_quality > 0.0:
        qual1 = compute_corner_quality(img1, esquinas1)
        qual2 = compute_corner_quality(img2, esquinas2)
        good_ids = []
        for mid in sorted(set(esquinas1.keys()) & set(esquinas2.keys())):
            mq1 = min(qual1.get(mid, [0.0]))
            mq2 = min(qual2.get(mid, [0.0]))
            if mq1 >= min_corner_quality and mq2 >= min_corner_quality:
                good_ids.append(mid)
        ids_comunes = good_ids
    else:
        ids_comunes = sorted(set(esquinas1.keys()) & set(esquinas2.keys()))

    if not ids_comunes:
        raise RuntimeError("No se encontraron marcadores ArUco comunes entre las dos imágenes.")

    pts1_list = []
    pts2_list = []
    pts_marker_ids = []
    for mid in ids_comunes:
        c1 = esquinas1[mid]
        c2 = esquinas2[mid]
        c2_aligned = alinear_esquinas_cyclic(c1, c2)
        pts1_list.append(c1)
        pts2_list.append(c2_aligned)
        pts_marker_ids.extend([mid] * 4)

    pts1 = np.vstack(pts1_list)
    pts2 = np.vstack(pts2_list)
    pts_marker_ids = np.array(pts_marker_ids, dtype=int)

    config.info(f"[INFO] Marcadores ArUco comunes: {ids_comunes}")
    config.info(f"[INFO] Total de correspondencias: {len(pts1)} puntos")

    return pts1, pts2, ids_comunes, pts_marker_ids


# ---------------------------------------------------------------------------
# 2. Algoritmo de los 8 puntos normalizado (NumPy puro)
# ---------------------------------------------------------------------------

def normalizar_puntos(pts):
    """
    Normalización de Hartley: traslada el centroide al origen y escala
    para que la distancia media al origen sea sqrt(2).

    Parámetros
    ----------
    pts : np.ndarray (Nx2)
        Puntos 2D.

    Retorna
    -------
    pts_norm : np.ndarray (Nx2)
        Puntos normalizados.
    T : np.ndarray (3x3)
        Matriz de normalización.
    """
    centroide = np.mean(pts, axis=0)
    pts_centrados = pts - centroide

    dist_media = np.mean(np.sqrt(np.sum(pts_centrados ** 2, axis=1)))
    if dist_media < 1e-10:
        dist_media = 1e-10

    s = np.sqrt(2.0) / dist_media

    T = np.array([
        [s,  0, -s * centroide[0]],
        [0,  s, -s * centroide[1]],
        [0,  0,                 1]
    ], dtype=np.float64)

    pts_hom = np.column_stack([pts, np.ones(len(pts))])
    pts_norm_hom = (T @ pts_hom.T).T
    pts_norm = pts_norm_hom[:, :2]

    return pts_norm, T


def eight_point_algorithm(pts1, pts2):
    """
    Algoritmo de los 8 puntos normalizado para estimar la Matriz Fundamental.

    Implementación completa en NumPy:
    1. Normalizar puntos.
    2. Construir el sistema lineal Af = 0.
    3. Resolver por SVD.
    4. Forzar rango 2 en F.
    5. Des-normalizar.

    Parámetros
    ----------
    pts1, pts2 : np.ndarray (Nx2)
        Al menos 8 correspondencias.

    Retorna
    -------
    F : np.ndarray (3x3)
        Matriz Fundamental estimada.
    """
    assert len(pts1) >= 8, "Se necesitan al menos 8 correspondencias."

    # Paso 1: normalizar
    pts1_n, T1 = normalizar_puntos(pts1)
    pts2_n, T2 = normalizar_puntos(pts2)

    n = len(pts1_n)
    x1, y1 = pts1_n[:, 0], pts1_n[:, 1]
    x2, y2 = pts2_n[:, 0], pts2_n[:, 1]

    # Paso 2: construir A  (cada fila: x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1)
    A = np.column_stack([
        x2 * x1,  x2 * y1,  x2,
        y2 * x1,  y2 * y1,  y2,
        x1,       y1,       np.ones(n)
    ])

    # Paso 3: resolver Af = 0 por SVD
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]                   # vector singular asociado al menor valor singular
    F_hat = f.reshape(3, 3)

    # Paso 4: forzar rango 2 (det(F) = 0)
    Uf, Sf, Vft = np.linalg.svd(F_hat)
    Sf[2] = 0.0                     # anular el menor valor singular
    F_norm = Uf @ np.diag(Sf) @ Vft

    # Paso 5: des-normalizar
    F = T2.T @ F_norm @ T1

    # Normalizar para que ||F|| = 1  (convención)
    F = F / np.linalg.norm(F)

    return F


# ---------------------------------------------------------------------------
# 3. RANSAC para Matriz Fundamental
# ---------------------------------------------------------------------------

def error_sampson(F, pts1, pts2):
    """
    Calcula el error de Sampson (distancia de primer orden) para cada
    correspondencia respecto a la Fundamental F.

    Parámetros
    ----------
    F : np.ndarray (3x3)
    pts1, pts2 : np.ndarray (Nx2)

    Retorna
    -------
    errores : np.ndarray (N,)
        Error de Sampson por correspondencia.
    """
    n = len(pts1)
    ones = np.ones((n, 1))
    p1 = np.hstack([pts1, ones])   # (N, 3)
    p2 = np.hstack([pts2, ones])

    # l2 = F @ p1^T   →  líneas epipolares en imagen 2
    Fp1 = (F @ p1.T).T            # (N, 3)
    # l1 = F^T @ p2^T →  líneas epipolares en imagen 1
    Ftp2 = (F.T @ p2.T).T         # (N, 3)

    # p2^T F p1  (escalar por correspondencia)
    numerador = np.sum(p2 * Fp1, axis=1) ** 2

    denominador = (Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 +
                   Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2)

    errores = numerador / (denominador + 1e-15)
    return errores


def ransac_fundamental(pts1, pts2,
                       umbral=2.0,
                       max_iter=5000,
                       confianza=0.999,
                       marker_ids=None):
    """
    Estimación robusta de la Matriz Fundamental usando OpenCV para máxima estabilidad.
    """
    F_mejor, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, umbral, confianza)
    if F_mejor is None:
        raise RuntimeError("OpenCV no logró encontrar un modelo válido de Matriz Fundamental.")
    
    mejor_mascara = mask.ravel().astype(bool)
    mejor_num_inliers = np.sum(mejor_mascara)
    n = len(pts1)
    
    config.info(f"[INFO] OpenCV RANSAC: {mejor_num_inliers}/{n} inliers "
        f"({100.0*mejor_num_inliers/n:.1f}%)")

    return F_mejor, mejor_mascara


# ---------------------------------------------------------------------------
# 4. Matriz Esencial
# ---------------------------------------------------------------------------

def calcular_esencial(F, K):
    """
    Calcula la Matriz Esencial de forma robusta.
    Aunque podemos usar E = K.T @ F @ K, usar la función interna
    de OpenCV si es posible garantiza mayor estabilidad numérica.
    Aquí mantenemos la definición matemática pura.
    """
    E_raw = K.T @ F @ K

    # SVD de E_raw
    U, S, Vt = np.linalg.svd(E_raw)

    # Forzar valores singulares: (σ, σ, 0) con σ = promedio de s1, s2
    sigma = (S[0] + S[1]) / 2.0
    S_corregidos = np.array([sigma, sigma, 0.0])

    E = U @ np.diag(S_corregidos) @ Vt

    # Normalizar
    E = E / np.linalg.norm(E) * np.sqrt(2.0)

    return E


# ---------------------------------------------------------------------------
# 5. Visualización de líneas epipolares
# ---------------------------------------------------------------------------

def dibujar_lineas_epipolares(img1, img2, F, pts1, pts2, num_lineas=15):
    """
    Dibuja líneas epipolares en ambas imágenes para visualización.

    Retorna
    -------
    vis1 : np.ndarray
        Imagen 1 con líneas epipolares de puntos de imagen 2.
    vis2 : np.ndarray
        Imagen 2 con líneas epipolares de puntos de imagen 1.
    """
    vis1 = img1.copy()
    vis2 = img2.copy()

    # Seleccionar un subconjunto (determinista para reproducibilidad)
    n = len(pts1)
    indices = list(range(0, n, max(1, n // num_lineas)))[:num_lineas]

    # Generar colores como tuplas de int (B,G,R) válidas para OpenCV
    colores = []
    for i in range(len(indices)):
        b = int((50 + i * 37) % 256)
        g = int((150 + i * 53) % 256)
        r = int((200 + i * 29) % 256)
        colores.append((b, g, r))

    for idx, color in zip(indices, colores):
        p1 = pts1[idx]
        p2 = pts2[idx]

        p1_h = np.array([p1[0], p1[1], 1.0])
        l2 = F @ p1_h
        _dibujar_linea_epipolar(vis2, l2, color)
        cv2.circle(vis2, (int(p2[0]), int(p2[1])), 6, color, -1)

        p2_h = np.array([p2[0], p2[1], 1.0])
        l1 = F.T @ p2_h
        _dibujar_linea_epipolar(vis1, l1, color)
        cv2.circle(vis1, (int(p1[0]), int(p1[1])), 6, color, -1)

    return vis1, vis2


def _dibujar_linea_epipolar(img, linea, color, grosor=2):
    """Dibuja una línea epipolar ax + by + c = 0 sobre la imagen."""
    a, b, c = linea
    h, w = img.shape[:2]
    if abs(b) > 1e-8:
        # x = 0
        y0 = int(-c / b)
        # x = w
        y1 = int(-(a * w + c) / b)
        cv2.line(img, (0, y0), (w, y1), color, grosor)
    elif abs(a) > 1e-8:
        x = int(-c / a)
        cv2.line(img, (x, 0), (x, h), color, grosor)


# ---------------------------------------------------------------------------
# Verificación de la restricción epipolar
# ---------------------------------------------------------------------------

def verificar_restriccion_epipolar(F, pts1, pts2):
    """
    Verifica que p2^T · F · p1 ≈ 0 para todas las correspondencias.

    Retorna
    -------
    errores : np.ndarray (N,)
        |p2^T F p1| por correspondencia.
    error_medio : float
        Promedio de los errores absolutos.
    """
    n = len(pts1)
    ones = np.ones((n, 1))
    p1_h = np.hstack([pts1, ones])
    p2_h = np.hstack([pts2, ones])

    errores = np.abs(np.sum(p2_h * (F @ p1_h.T).T, axis=1))
    error_medio = np.mean(errores)

    config.info(f"[INFO] Restricción epipolar:  error medio = {error_medio:.6e}, "
        f"max = {np.max(errores):.6e}")

    return errores, error_medio


# ---------------------------------------------------------------------------
# Ejecución directa (para pruebas)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    img1 = cv2.imread(os.path.join("FotosOriginales", "1.png"))
    img2 = cv2.imread(os.path.join("FotosOriginales", "2.png"))

    if img1 is None or img2 is None:
        print("[ERROR] No se pudieron cargar las imágenes del par estéreo.")
    else:
        pts1, pts2, ids = obtener_correspondencias_aruco(img1, img2)
        F, mascara = ransac_fundamental(pts1, pts2)
        print(f"\nMatriz Fundamental F:\n{F}")
        verificar_restriccion_epipolar(F, pts1[mascara], pts2[mascara])
