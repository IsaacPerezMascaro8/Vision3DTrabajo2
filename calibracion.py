"""
calibracion.py
==============
Calibración de cámara usando un patrón de tablero de ajedrez (checkerboard).

Detecta las esquinas internas del checkerboard en las imágenes de
calibración y usa cv2.calibrateCamera para obtener K, distorsión
y error de reproyección.

Configuración:
    CHECKERBOARD_SIZE : esquinas internas (columnas, filas).
                        Para un tablero de 8×6 cuadrados → (7, 5).
    SQUARE_SIZE_MM    : tamaño del lado de cada cuadrado en mm.
"""

import os
import glob
import numpy as np
import cv2


# =====================================================================
# CONFIGURACIÓN DEL CHECKERBOARD
# =====================================================================
# Esquinas internas: para un tablero de 8×6 cuadrados → (7, 5)
CHECKERBOARD_SIZE = (7, 5)

# Tamaño del lado de cada cuadrado en mm.
# >>> MEDIR CON REGLA Y AJUSTAR SI ES NECESARIO <<<
SQUARE_SIZE_MM = 30.0


# =====================================================================
# Funciones auxiliares
# =====================================================================

def generar_puntos_3d_checkerboard(size=CHECKERBOARD_SIZE,
                                    square_size=SQUARE_SIZE_MM):
    """
    Genera las coordenadas 3D de las esquinas internas del checkerboard.

    El patrón se sitúa en el plano Z=0, con origen en la primera esquina.

    Retorna
    -------
    objp : np.ndarray (N, 3)
        Coordenadas 3D de las N = size[0]*size[1] esquinas.
    """
    objp = np.zeros((size[0] * size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


# =====================================================================
# Pipeline de calibración
# =====================================================================

def calibrar_camara(directorio_imagenes, mostrar=False, target_size=None):
    """
    Calibra la cámara usando un patrón de tablero de ajedrez.

    Parámetros
    ----------
    directorio_imagenes : str
        Directorio con las imágenes de calibración (.jpg/.jpeg/.png).
    target_size : tuple (ancho, alto) | None
        Si se indica, solo se usarán imágenes con esta resolución.
        Útil cuando hay mezcla de orientaciones (vertical/horizontal).
    mostrar : bool
        Si True, muestra las detecciones en pantalla.

    Retorna
    -------
    K : np.ndarray (3x3)
        Matriz intrínseca de la cámara.
    dist : np.ndarray
        Coeficientes de distorsión (k1, k2, p1, p2, k3).
    rvecs : list[np.ndarray]
        Vectores de rotación por imagen.
    tvecs : list[np.ndarray]
        Vectores de translación por imagen.
    error_medio : float
        Error medio de reproyección en píxeles.
    image_size : tuple
        (ancho, alto) de las imágenes.
    """
    # Buscar imágenes
    extensiones = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    rutas = []
    for ext in extensiones:
        rutas.extend(glob.glob(os.path.join(directorio_imagenes, ext)))
    rutas.sort()

    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron imágenes en: {directorio_imagenes}")

    # Puntos 3D del patrón (iguales para todas las imágenes)
    objp = generar_puntos_3d_checkerboard()

    # Criterio de terminación para cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)

    all_obj_pts = []
    all_img_pts = []
    image_size = target_size
    imagenes_usadas = 0

    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is None:
            continue

        current_size = (img.shape[1], img.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            continue

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar esquinas del checkerboard
        ret, corners = cv2.findChessboardCorners(
            gris, CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            continue

        # Refinar esquinas a precisión sub-píxel
        corners_refined = cv2.cornerSubPix(
            gris, corners, (11, 11), (-1, -1), criteria)

        all_obj_pts.append(objp)
        all_img_pts.append(corners_refined)
        imagenes_usadas += 1

        if mostrar:
            img_vis = img.copy()
            cv2.drawChessboardCorners(
                img_vis, CHECKERBOARD_SIZE, corners_refined, ret)
            h_show = 800
            ratio = h_show / img_vis.shape[0]
            img_vis = cv2.resize(img_vis, None, fx=ratio, fy=ratio)
            cv2.imshow("Checkerboard detectado", img_vis)
            cv2.waitKey(200)

    if mostrar:
        cv2.destroyAllWindows()

    if imagenes_usadas < 3:
        raise RuntimeError(
            f"Solo {imagenes_usadas} imágenes válidas. Se necesitan >= 3.")

    # Calibrar
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts, all_img_pts, image_size, None, None)

    # Error medio de reproyección
    error_total = 0.0
    total_puntos = 0
    for i in range(len(all_obj_pts)):
        reproyectados, _ = cv2.projectPoints(
            all_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(all_img_pts[i], reproyectados, cv2.NORM_L2)
        n = len(all_obj_pts[i])
        error_total += error ** 2
        total_puntos += n

    error_medio = np.sqrt(error_total / total_puntos)

    return K, dist, rvecs, tvecs, error_medio, image_size


# =====================================================================
# Corrección de distorsión
# =====================================================================

def corregir_distorsion(imagen, K, dist):
    """
    Corrige la distorsión radial/tangencial de una imagen.

    Se usa cv2.undistort con la misma K como new camera matrix,
    de modo que la K original sigue siendo válida tras la corrección.

    Retorna
    -------
    imagen_corregida : np.ndarray
    """
    # By default keep same K as new camera matrix (no cropping).
    imagen_corregida = cv2.undistort(imagen, K, dist, None, K)
    return imagen_corregida


def cargar_y_corregir(ruta_imagen, K, dist):
    """
    Carga una imagen desde disco y le aplica corrección de distorsión.

    Retorna
    -------
    imagen_corregida : np.ndarray (BGR)
    """
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {ruta_imagen}")
    # Compute optimal new camera matrix to avoid cropping important regions
    # Use alpha=1 to keep all pixels (may introduce black borders)
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1)
    img_corr = cv2.undistort(img, K, dist, None, newK)
    return img_corr, newK


# =====================================================================
# Ejecución directa (prueba)
# =====================================================================
if __name__ == "__main__":
    DIR_CALIBRACION = "FotosCalibracion"
    K, dist, rvecs, tvecs, error, img_size = calibrar_camara(
        DIR_CALIBRACION, mostrar=False)
