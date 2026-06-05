import os
import cv2
import numpy as np
import config
import matplotlib.pyplot as plt


def calcular_error_vertical(pts1, pts2, pts1_rect, pts2_rect):
    """Calcula y muestra la media de la diferencia absoluta en coordenada Y
    antes y después de la rectificación.

    Parameters
    ----------
    pts1, pts2 : ndarray de forma (N, 2)
        Puntos originales detectados en la primera y segunda imagen.
    pts1_rect, pts2_rect : ndarray de forma (N, 2)
        Puntos mapeados al plano rectificado (salida de cv2.undistortPoints).
    """
    # Diferencia vertical antes de rectificar
    diff_y_before = np.abs(pts1[:, 1] - pts2[:, 1])
    mean_before = np.mean(diff_y_before)
    # Diferencia vertical después de rectificar
    diff_y_after = np.abs(pts1_rect[:, 1] - pts2_rect[:, 1])
    mean_after = np.mean(diff_y_after)
    config.info(f"[INFO] Media diferencia Y antes rectificacion: {mean_before:.4f}")
    config.info(f"[INFO] Media diferencia Y despues rectificacion: {mean_after:.4f}")
    return mean_before, mean_after


def calcular_mapa_disparidad(img_rect1, img_rect2):
    """Calcula el mapa de disparidad densa usando StereoSGBM.

    Returns
    -------
    disparity_raw : ndarray (float32)
        Mapa de disparidad sin normalizar (en unidades de píxel).
    disparity_color : ndarray (uint8, 3 canales)
        Versión coloreada para visualización.
    """
    # Convertir a escala de grises
    gray1 = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_rect2, cv2.COLOR_BGR2GRAY)

    # Configuración optimizada para eliminar ruido en superficies planas
    window_size = 5
    num_disp = 16 * 8  # rango de disparidad ampliado

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Cálculo (OpenCV devuelve valores × 16)
    disparidad_cruda = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Filtrar valores inválidos (OpenCV asigna valores negativos a píxeles sin correspondencia)
    mask = disparidad_cruda > 0
    disparidad_filtrada = np.zeros_like(disparidad_cruda)
    disparidad_filtrada[mask] = disparidad_cruda[mask]

    # Normalización segura para el colormap (escala 0-255)
    disp_normalizada = cv2.normalize(disparidad_filtrada, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_normalizada)

    # Aplicar mapa de color
    mapa_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)

    return disparidad_cruda, mapa_color


def extraer_perfiles_disparidad(disparidad_cruda, filas_y):
    """Genera una gráfica de los perfiles de disparidad a lo largo de filas específicas.

    Parameters
    ----------
    disparidad_cruda : ndarray (float32)
        Mapa de disparidad sin normalizar.
    filas_y : list[int]
        Coordenadas Y de las filas a graficar (ej. [h//3, 2*h//3]).
    """
    plt.figure(figsize=(10, 4))
    for y in filas_y:
        if 0 <= y < disparidad_cruda.shape[0]:
            perfil = disparidad_cruda[y, :]
            plt.plot(perfil, label=f"Fila Y={y}")
    plt.title("Perfiles de disparidad por fila")
    plt.xlabel("Columna (pixel)")
    plt.ylabel("Disparidad (pixel)")
    plt.legend()
    plt.grid(True)
    out_dir = config.ensure_output_dir("output")
    out_path = os.path.join(out_dir, "disparity_profiles.png")
    if config.SHOW_GUI:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        config.info(f"[INFO] Perfiles de disparidad guardados en: {out_path}")
    plt.close()
    return out_path
