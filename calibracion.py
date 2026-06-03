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

def extraer_esquinas(rutas, objp, criteria, target_size, mostrar):
    all_obj_pts, all_img_pts = [], []
    image_size = target_size
    imagenes_usadas = 0

    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is None: continue
        current_size = (img.shape[1], img.shape[0])
        if image_size is None: image_size = current_size
        elif current_size != image_size: continue

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gris, CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret: continue

        corners_refined = cv2.cornerSubPix(gris, corners, (11, 11), (-1, -1), criteria)
        all_obj_pts.append(objp)
        all_img_pts.append(corners_refined)
        imagenes_usadas += 1

        if mostrar:
            img_vis = img.copy()
            cv2.drawChessboardCorners(img_vis, CHECKERBOARD_SIZE, corners_refined, ret)
            ratio = 800 / img_vis.shape[0]
            cv2.imshow("Checkerboard detectado", cv2.resize(img_vis, None, fx=ratio, fy=ratio))
            cv2.waitKey(200)

    if mostrar: cv2.destroyAllWindows()
    return all_obj_pts, all_img_pts, image_size, imagenes_usadas

def calibrar_camara(directorio_imagenes, mostrar=False, target_size=None):
    """Calibra la cámara usando un patrón de tablero de ajedrez."""
    rutas = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
        rutas.extend(glob.glob(os.path.join(directorio_imagenes, ext)))
    rutas.sort()
    if not rutas: raise FileNotFoundError(f"No se encontraron imágenes en: {directorio_imagenes}")

    objp = generar_puntos_3d_checkerboard()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    all_obj_pts, all_img_pts, image_size, img_usadas = extraer_esquinas(rutas, objp, criteria, target_size, mostrar)

    if img_usadas < 3: raise RuntimeError(f"Solo {img_usadas} imágenes válidas. Se necesitan >= 3.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_pts, all_img_pts, image_size, None, None)

    error_total, total_puntos = 0.0, 0
    for i in range(len(all_obj_pts)):
        reproyectados, _ = cv2.projectPoints(all_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(all_img_pts[i], reproyectados, cv2.NORM_L2)
        error_total += error ** 2
        total_puntos += len(all_obj_pts[i])

    error_medio = np.sqrt(error_total / total_puntos)
    return K, dist, rvecs, tvecs, error_medio, image_size





# =====================================================================
# Ejecución directa (prueba)
# =====================================================================
if __name__ == "__main__":
    DIR_CALIBRACION = "FotosCalibracion"
    K, dist, rvecs, tvecs, error, img_size = calibrar_camara(
        DIR_CALIBRACION, mostrar=False)
