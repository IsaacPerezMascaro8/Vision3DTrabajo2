"""
Grid Search visual de hiperparámetros para Plane Sweeping.

Ejecuta plane_sweep_stereo con 10 configuraciones estratégicas y guarda
los mapas de color resultantes en tests_parametros/ para comparación visual.

Uso:
    python grid_search.py
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

# Matriz intrínseca (idéntica a main.py)
K = np.array([[3954.809289, 0.0, 2133.663739],
              [0.0, 3956.467881, 2889.190270],
              [0.0, 0.0, 1.0]])

# ── 10 configuraciones estratégicas ──────────────────────────────────────────
CONFIGS = [
    {"nombre": "01_Base",              "box": 7,  "umb": 1.25, "k": 5},
    {"nombre": "02_Umbral_relajado",   "box": 7,  "umb": 1.8,  "k": 5},
    {"nombre": "03_Umbral_muy_relaj",  "box": 7,  "umb": 2.5,  "k": 5},
    {"nombre": "04_Tapacubos_medio",   "box": 7,  "umb": 1.25, "k": 9},
    {"nombre": "05_Tapacubos_gigante", "box": 7,  "umb": 1.25, "k": 15},
    {"nombre": "06_Red_pesca_grande",  "box": 11, "umb": 1.25, "k": 5},
    {"nombre": "07_Red_pesca_gigante", "box": 15, "umb": 1.25, "k": 5},
    {"nombre": "08_Equilibrio_1",      "box": 11, "umb": 1.5,  "k": 9},
    {"nombre": "09_Equilibrio_2",      "box": 11, "umb": 2.0,  "k": 11},
    {"nombre": "10_El_Tanque",         "box": 15, "umb": 2.0,  "k": 15},
]


def main():
    config.set_show_gui(False)

    outdir = "tests_parametros"
    os.makedirs(outdir, exist_ok=True)

    # 1. Cargar imágenes
    img1 = cv2.imread(os.path.join("FotosOriginales", "1.png"))
    img2 = cv2.imread(os.path.join("FotosOriginales", "2.png"))
    if img1 is None or img2 is None:
        sys.exit("Error: No se pudieron cargar las imágenes.")

    # 2. Pipeline geométrico (mismo que main.py)
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

    # 4. Grid Search
    config.separator("GRID SEARCH DE HIPERPARÁMETROS")
    total = len(CONFIGS)

    for i, cfg in enumerate(CONFIGS):
        nombre = cfg["nombre"]
        box = cfg["box"]
        umb = cfg["umb"]
        k = cfg["k"]

        config.info(f"\n[GRID {i+1}/{total}] {nombre}: box={box}, umb={umb}, k={k}")

        _, mapa_color = plane_sweep_stereo(
            img_rect1, img_rect2,
            box_size=box,
            umbral_mult=umb,
            morph_k=k
        )

        filename = f"test_box{box}_umb{umb}_k{k}.jpg"
        filepath = os.path.join(outdir, filename)
        cv2.imwrite(filepath, mapa_color)
        config.info(f"[GRID {i+1}/{total}] Guardado: {filepath}")

    config.separator("GRID SEARCH COMPLETADO")
    config.info(f"Resultados en: {outdir}/")
    config.info(f"Total: {total} configuraciones testeadas.")


if __name__ == "__main__":
    main()
