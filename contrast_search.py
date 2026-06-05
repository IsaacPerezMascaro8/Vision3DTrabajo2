"""
Grid Search visual de hiperparámetros de recorte (clip) para Plane Sweeping.

Ejecuta plane_sweep_stereo con 6 configuraciones estratégicas de np.clip
para maximizar el contraste de los objetos centrales y guarda los mapas
resultantes en tests_contraste/.

Uso:
    python contrast_search.py
"""
import os
import sys
import numpy as np
import cv2
import config

from geometria_epipolar import obtener_correspondencias_aruco, ransac_fundamental, calcular_esencial
from reconstruccion import seleccionar_pose
from rectificacion import rectificar_par_estereo
from plane_sweeping import plane_sweep_stereo

# Matriz intrínseca
K = np.array([[3954.809289, 0.0, 2133.663739],
              [0.0, 3956.467881, 2889.190270],
              [0.0, 0.0, 1.0]])

# ── 6 configuraciones de contraste ──────────────────────────────────────────
CONFIGS = [
    {"nombre": "01_Rango_Completo",        "cmin": 0,  "cmax": 128},
    {"nombre": "02_Corte_Suave",           "cmin": 10, "cmax": 110},
    {"nombre": "03_Corte_Medio_Centro",    "cmin": 20, "cmax": 90},
    {"nombre": "04_Corte_Agresivo_Max",    "cmin": 30, "cmax": 80},
    {"nombre": "05_Cortar_Fondo_Infinito", "cmin": 0,  "cmax": 90},
    {"nombre": "06_Cortar_Ruido_Primer",   "cmin": 25, "cmax": 128},
]


def main():
    config.set_show_gui(False)

    outdir = "tests_contraste"
    os.makedirs(outdir, exist_ok=True)

    # 1. Cargar imágenes
    img1 = cv2.imread(os.path.join("FotosOriginales", "1.png"))
    img2 = cv2.imread(os.path.join("FotosOriginales", "2.png"))
    if img1 is None or img2 is None:
        sys.exit("Error: No se pudieron cargar las imágenes.")

    # 2. Pipeline geométrico
    pts1, pts2, ids, p_ids = obtener_correspondencias_aruco(img1, img2)
    F, mask = ransac_fundamental(pts1, pts2, umbral=2.0, max_iter=5000, marker_ids=p_ids)
    p1_in, p2_in = pts1[mask], pts2[mask]
    E = calcular_esencial(F, K)
    R, t, X, _ = seleccionar_pose(E, K, p1_in, p2_in)

    # Escala métrica
    TAMANO_ARUCO_M = 0.12
    from reconstruccion import triangular_punto
    P1_temp = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_temp = K @ np.hstack([R, t])
    X_corner0 = triangular_punto(P1_temp, P2_temp, pts1[0], pts2[0])
    X_corner1 = triangular_punto(P1_temp, P2_temp, pts1[1], pts2[1])
    distancia_3d = np.linalg.norm(X_corner0 - X_corner1)
    factor_escala = TAMANO_ARUCO_M / distancia_3d
    t = t * factor_escala

    # 3. Rectificación
    img_rect1, img_rect2, _, _, _, _ = rectificar_par_estereo(img1, img2, K, R, t)

    # 4. Contrast Search
    config.separator("GRID SEARCH DE CONTRASTE")
    total = len(CONFIGS)

    for i, cfg in enumerate(CONFIGS):
        nombre = cfg["nombre"]
        cmin = cfg["cmin"]
        cmax = cfg["cmax"]

        config.info(f"\n[CONTRASTE {i+1}/{total}] {nombre}: clip_min={cmin}, clip_max={cmax}")

        # Ejecutamos con la configuración base ganadora de agujeros
        _, mapa_color = plane_sweep_stereo(
            img_rect1, img_rect2,
            box_size=7,
            umbral_mult=2.5,
            morph_k=5,
            clip_min=cmin,
            clip_max=cmax
        )

        filename = f"test_clip_min{cmin}_max{cmax}.jpg"
        filepath = os.path.join(outdir, filename)
        cv2.imwrite(filepath, mapa_color)
        config.info(f"[CONTRASTE {i+1}/{total}] Guardado: {filepath}")

    config.separator("BÚSQUEDA DE CONTRASTE COMPLETADA")
    config.info(f"Resultados guardados en la carpeta: {outdir}/")


if __name__ == "__main__":
    main()
