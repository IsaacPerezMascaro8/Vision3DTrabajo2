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

def detectar_aruco(imagen, diccionario=cv2.aruco.DICT_4X4_50, refine_corners=False):
    """
    Detecta marcadores ArUco en una imagen y devuelve las esquinas
    organizadas por ID.

    Parámetros
    ----------
    imagen : np.ndarray
        Imagen BGR.
    diccionario : int
        Tipo de diccionario ArUco.

    Retorna
    -------
    esquinas_por_id : dict
        {marker_id: np.ndarray de forma (4, 2)} con las 4 esquinas.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(diccionario)
    parametros = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parametros)

    corners, ids, _ = detector.detectMarkers(imagen)

    esquinas_por_id = {}
    if ids is not None:
        # Optionally refine corners with subpixel precision
        if refine_corners:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            # prepare all corners in the format required by cornerSubPix: (N,1,2)
            all_corners = []
            id_list = []
            for i, marker_id in enumerate(ids.ravel()):
                c = corners[i].reshape(4, 2)
                for corner in c:
                    all_corners.append(corner)
                id_list.append(int(marker_id))

            if len(all_corners) > 0:
                pts = np.array(all_corners, dtype=np.float32).reshape(-1, 1, 2)
                # termination criteria and window sizes are standard choices
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
                cv2.cornerSubPix(gray, pts, winSize=(5,5), zeroZone=(-1,-1), criteria=criteria)
                pts = pts.reshape(-1, 2)

                # rebuild per-marker arrays
                k = 0
                for i, marker_id in enumerate(ids.ravel()):
                    refined = pts[k:k+4]
                    esquinas_por_id[int(marker_id)] = refined.copy()
                    k += 4
        else:
            for i, marker_id in enumerate(ids.ravel()):
                esquinas_por_id[int(marker_id)] = corners[i].reshape(4, 2)

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
    Estimación robusta de la Matriz Fundamental con RANSAC.

    Parámetros
    ----------
    pts1, pts2 : np.ndarray (Nx2)
        Correspondencias (mínimo 8).
    umbral : float
        Umbral de error de Sampson para considerar un inlier (en píxeles).
    max_iter : int
        Número máximo de iteraciones.
    confianza : float
        Nivel de confianza deseado para detención temprana.

    Retorna
    -------
    F_mejor : np.ndarray (3x3)
        Matriz Fundamental estimada con los mejores inliers.
    mascara_inliers : np.ndarray (N,) bool
        Máscara de inliers.
    """
    n = len(pts1)
    assert n >= 8, f"Se necesitan al menos 8 correspondencias, se tienen {n}."

    mejor_num_inliers = 0
    mejor_mascara = np.zeros(n, dtype=bool)
    F_mejor = None

    # Umbral al cuadrado para Sampson (la función error_sampson devuelve
    # el valor que corresponde a la distancia al cuadrado en píxeles)
    umbral_sq = umbral ** 2

    # RNG reproducible (estilo del código de homografía)
    rng = np.random.default_rng(42)

    iteraciones_adaptativas = max_iter

    for it in range(max_iter):
        if it >= iteraciones_adaptativas:
            break

        # Seleccionar 8 puntos al azar
        # Avoid degenerate samples: ensure sampled indices come from at least
        # min_distinct_markers different markers (if marker_ids provided).
        # Prefer 6 distinct markers when available (better conditioning).
        max_sample_attempts = 100
        if marker_ids is None:
            indices = rng.choice(n, 8, replace=False)
        else:
            unique_markers = np.unique(marker_ids)
            if len(unique_markers) >= 6:
                required_distinct = 6
            else:
                required_distinct = max(4, min(6, len(unique_markers)))

            for attempt in range(max_sample_attempts):
                indices = rng.choice(n, 8, replace=False)
                sampled_markers = np.unique(marker_ids[indices])
                if len(sampled_markers) >= required_distinct:
                    break
            else:
                # fallback: accept last sample
                indices = rng.choice(n, 8, replace=False)

        try:
            F_candidato = eight_point_algorithm(pts1[indices], pts2[indices])
        except Exception:
            continue

        # Calcular errores
        errores = error_sampson(F_candidato, pts1, pts2)
        mascara = errores < umbral_sq
        num_inliers = np.sum(mascara)

        if num_inliers > mejor_num_inliers:
            mejor_num_inliers = num_inliers
            mejor_mascara = mascara.copy()
            F_mejor = F_candidato.copy()

            # Actualización adaptativa del número de iteraciones (LO-RANSAC style)
            ratio_inliers = num_inliers / n
            if ratio_inliers > 0.0 and ratio_inliers < 1.0:
                num_necesario = (np.log(1.0 - confianza) /
                                 np.log(1.0 - ratio_inliers ** 8))
                iteraciones_adaptativas = min(max_iter,
                                              int(num_necesario) + 1)

    if F_mejor is None:
        raise RuntimeError("RANSAC no logró encontrar un modelo válido.")

    # Re-estimar F con todos los inliers
    inlier_pts1 = pts1[mejor_mascara]
    inlier_pts2 = pts2[mejor_mascara]
    if len(inlier_pts1) >= 8:
        F_mejor = eight_point_algorithm(inlier_pts1, inlier_pts2)

    config.info(f"[INFO] RANSAC: {mejor_num_inliers}/{n} inliers "
        f"({100.0*mejor_num_inliers/n:.1f}%), "
        f"iteraciones usadas: {min(it+1, iteraciones_adaptativas)}")

    # Re-estimar F con todos los inliers encontrados (estilo RANSAC DLT)
    if F_mejor is None:
        raise RuntimeError("RANSAC no logró encontrar un modelo válido.")

    inlier_pts1 = pts1[mejor_mascara]
    inlier_pts2 = pts2[mejor_mascara]
    if len(inlier_pts1) >= 8:
        F_mejor = eight_point_algorithm(inlier_pts1, inlier_pts2)

    config.info(f"[INFO] RANSAC: {mejor_num_inliers}/{n} inliers "
        f"({100.0*mejor_num_inliers/n:.1f}%), "
        f"iteraciones usadas: {min(it+1, iteraciones_adaptativas)}")

    return F_mejor, mejor_mascara


# ---------------------------------------------------------------------------
# 4. Matriz Esencial
# ---------------------------------------------------------------------------

def calcular_esencial(F, K):
    """
    Calcula la Matriz Esencial a partir de F y K:
        E = K^T · F · K

    Luego fuerza los valores singulares a la forma diag(σ, σ, 0) donde
    σ = (s1 + s2) / 2.

    Parámetros
    ----------
    F : np.ndarray (3x3)
        Matriz Fundamental.
    K : np.ndarray (3x3)
        Matriz intrínseca de la cámara.

    Retorna
    -------
    E : np.ndarray (3x3)
        Matriz Esencial con valores singulares forzados.
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

    config.info(f"[INFO] Valores singulares de E (crudos):    {S}")
    config.info(f"[INFO] Valores singulares de E (forzados):  {S_corregidos}")

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
