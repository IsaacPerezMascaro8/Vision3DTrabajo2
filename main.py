"""
main.py
=======
Pipeline principal de Visión 3D.

Orquesta los tres módulos:
  1. calibracion.py     → Calibración de la cámara (K, distorsión).
  2. geometria_epipolar.py → Correspondencias ArUco, F, E.
  3. reconstruccion.py  → Descomposición de E, triangulación, quiralidad.

Uso:
    python main.py
"""

import os
import sys
import argparse
import numpy as np
import cv2
import config

# Importar módulos del proyecto
from calibracion import cargar_y_corregir
from geometria_epipolar import (
    obtener_correspondencias_aruco,
    ransac_fundamental,
    calcular_esencial,
    dibujar_lineas_epipolares,
    verificar_restriccion_epipolar,
)
from reconstruccion import (
    seleccionar_pose,
    error_reproyeccion,
    visualizar_reconstruccion,
    bundle_adjustment,
)


# ---------------------------------------------------------------------------
# Rutas del dataset
# ---------------------------------------------------------------------------
DIR_PAR_ESTEREO = "FotosOriginales"
IMG1_PATH       = os.path.join(DIR_PAR_ESTEREO, "1.png")
IMG2_PATH       = os.path.join(DIR_PAR_ESTEREO, "2.png")

# ---------------------------------------------------------------------------
# Parámetros de calibración (obtenidos con calibrar.py)
# Para recalibrar ejecutar:  python3 calibrar.py
# ---------------------------------------------------------------------------
K = np.array([
    [3954.8092889939, 0.0000000000, 2133.6637390706],
    [0.0000000000, 3956.4678808372, 2889.1902696194],
    [0.0000000000, 0.0000000000, 1.0000000000],
])

dist = np.array([0.2768972535, -2.1265108290, -0.0019064616, 0.0003102212, 4.8286872005])

ERROR_CALIB = 0.9395  # px — error medio de reproyección de la calibración


def separador(titulo):
    """Delegar al separador centralizado (config)."""
    config.separator(titulo)


def main():
    """Pipeline completo de visión 3D."""

    np.set_printoptions(precision=6, suppress=True)

    # Argumentos simples para controlar salida
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--quiet', action='store_true', help='Silenciar mensajes informativos')
    parser.add_argument('--no-gui', action='store_true', help='No abrir ventanas GUI; guardar salidas en output/')
    parser.add_argument('--outdir', type=str, default='output', help='Directorio donde guardar resultados cuando --no-gui')
    parser.add_argument('--refine-corners', action='store_true', help='Refinar esquinas ArUco con cornerSubPix antes de usar correspondencias')
    parser.add_argument('--ba-nfev', type=int, default=2000, help='Máximo de evaluaciones para bundle adjustment')
    parser.add_argument('--min-corner-quality', type=float, default=0.0, help='Umbral relativo (0..inf) para filtrar marcadores por calidad de esquina')
    parser.add_argument('--ransac-threshold', type=float, default=2.0, help='Umbral Sampson (px) para RANSAC')
    args, _ = parser.parse_known_args()
    parser.add_argument('--filter-reproj', type=float, default=30.0, help='Umbral (px) para filtrar puntos por error de reproyección tras triangulación')
    args, _ = parser.parse_known_args()

    if args.quiet:
        config.set_verbose(False)
    if args.no_gui:
        config.set_show_gui(False)
    outdir = args.outdir
    if not config.SHOW_GUI:
        outdir = config.ensure_output_dir(outdir)

    # ===================================================================
    # PASO 1: Parámetros de calibración
    # ===================================================================
    separador("PASO 1: CALIBRACIÓN DE LA CÁMARA (precalculada)")

    error_calib = ERROR_CALIB

    config.info(f"  → Matriz intrínseca K:")
    config.info(f"      fx = {K[0,0]:.2f} px")
    config.info(f"      fy = {K[1,1]:.2f} px")
    config.info(f"      cx = {K[0,2]:.2f} px")
    config.info(f"      cy = {K[1,2]:.2f} px")
    config.info(f"  → Error de reproyección:     {error_calib:.4f} px")
    config.info(f"  → (Para recalibrar: python3 calibrar.py)")

    # ===================================================================
    # PASO 2: Carga y corrección de distorsión del par estéreo
    # ===================================================================
    separador("PASO 2: CORRECCIÓN DE DISTORSIÓN DEL PAR ESTÉREO")

    if not os.path.isfile(IMG1_PATH) or not os.path.isfile(IMG2_PATH):
        print(f"[ERROR] No se encontraron las imágenes del par estéreo:")
        print(f"        {IMG1_PATH}")
        print(f"        {IMG2_PATH}")
        sys.exit(1)

    # keep originals for distortion comparison and rectification
    img1_orig = cv2.imread(IMG1_PATH)
    img2_orig = cv2.imread(IMG2_PATH)

    img1_corr, K_opt = cargar_y_corregir(IMG1_PATH, K, dist)
    img2_corr, _ = cargar_y_corregir(IMG2_PATH, K, dist)
    img_size = (img1_corr.shape[1], img1_corr.shape[0])

    config.info(f"  → Imagen 1 corregida: {img1_corr.shape[1]}x{img1_corr.shape[0]}")
    config.info(f"  → Imagen 2 corregida: {img2_corr.shape[1]}x{img2_corr.shape[0]}")

    # Mostrar o guardar imágenes corregidas según configuración
    def mostrar_img(titulo, img, max_h=800, save_path=None):
        if config.SHOW_GUI:
            ratio = max_h / img.shape[0]
            img_show = cv2.resize(img, None, fx=ratio, fy=ratio)
            cv2.imshow(titulo, img_show)
        else:
            outdir = config.ensure_output_dir('output')
            if save_path is None:
                save_path = os.path.join(outdir, f"{titulo.replace(' ', '_')}.png")
            cv2.imwrite(save_path, img)
            config.info(f"  → Imagen guardada: {save_path}")

    if config.SHOW_GUI:
        mostrar_img("Imagen 1 corregida", img1_corr)
        mostrar_img("Imagen 2 corregida", img2_corr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        mostrar_img('Imagen_1_corregida', img1_corr)
        mostrar_img('Imagen_2_corregida', img2_corr)

    # ===================================================================
    # PASO 3: Detección de correspondencias ArUco
    # ===================================================================
    separador("PASO 3: CORRESPONDENCIAS ArUco")

    pts1, pts2, ids_comunes, pts_marker_ids = obtener_correspondencias_aruco(
        img1_corr, img2_corr, refine_corners=args.refine_corners, min_corner_quality=args.min_corner_quality)

    config.info(f"  → IDs de marcadores comunes: {ids_comunes}")
    config.info(f"  → Número de correspondencias: {len(pts1)}")

    # ===================================================================
    # PASO 4: Estimación de la Matriz Fundamental con RANSAC
    # ===================================================================
    separador("PASO 4: MATRIZ FUNDAMENTAL (F) — RANSAC + 8 PUNTOS")

    F, mascara = ransac_fundamental(pts1, pts2, umbral=args.ransac_threshold, max_iter=5000, marker_ids=pts_marker_ids)

    pts1_inliers = pts1[mascara]
    pts2_inliers = pts2[mascara]

    config.info(f"\n  → Inliers: {np.sum(mascara)}/{len(pts1)}")
    config.info(f"  → Matriz Fundamental F:\n{F}")

    # Verificar restricción epipolar
    errores_epi, error_epi_medio = verificar_restriccion_epipolar(
        F, pts1_inliers, pts2_inliers)

    # Dibujar líneas epipolares
    vis1, vis2 = dibujar_lineas_epipolares(
        img1_corr, img2_corr, F, pts1_inliers, pts2_inliers)
    if config.SHOW_GUI:
        mostrar_img("Epipolares - Imagen 1", vis1)
        mostrar_img("Epipolares - Imagen 2", vis2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        config.info("  → Líneas epipolares mostradas.")
    else:
        outdir = config.ensure_output_dir('output')
        mostrar_img('Epipolares_Imagen_1', vis1)
        mostrar_img('Epipolares_Imagen_2', vis2)

    # ===================================================================
    # PASO 5: Matriz Esencial (E)
    # ===================================================================
    separador("PASO 5: MATRIZ ESENCIAL (E)")

    E = calcular_esencial(F, K)
    config.info(f"\n  → Matriz Esencial E:\n{E}")

    # ===================================================================
    # PASO 6: Descomposición de E y selección de pose
    # ===================================================================
    separador("PASO 6: DESCOMPOSICIÓN DE E — TRIANGULACIÓN Y QUIRALIDAD")

    R, t, puntos_3d, idx_pose = seleccionar_pose(
        E, K, pts1_inliers, pts2_inliers)

    # ===================================================================
    # PASO 7: Error de reproyección post-triangulación
    # ===================================================================
    separador("PASO 7: ERROR DE REPROYECCIÓN")

    err_cam1, err_cam2 = error_reproyeccion(
        K, R, t, puntos_3d, pts1_inliers, pts2_inliers)

    # Intentar bundle adjustment para refinar R,t y puntos 3D (si hay suficientes puntos)
    if len(puntos_3d) >= 4:
        # convertir R a rvec
        rvec_init = cv2.Rodrigues(R)[0].ravel()
        try:
            rvec_opt, t_opt, X_opt = bundle_adjustment(K, rvec_init, t, puntos_3d, pts1_inliers, pts2_inliers, max_nfev=args.ba_nfev)
            R_opt = cv2.Rodrigues(rvec_opt)[0]
            config.info('[INFO] Bundle adjustment completed. Calculating post-BA reprojection error...')
            err1_ba, err2_ba = error_reproyeccion(K, R_opt, t_opt, X_opt, pts1_inliers, pts2_inliers)

            # Guardar resultados BA
            outdir = config.ensure_output_dir('output')
            results_ba = {
                'error_reproyeccion_cam1_ba_px': float(err1_ba),
                'error_reproyeccion_cam2_ba_px': float(err2_ba)
            }
            config.save_json(results_ba, os.path.join(outdir, 'results_ba.json'))
            config.info(f"[INFO] BA results saved: {os.path.join(outdir, 'results_ba.json')}")

            # overlays BA
            n_pts = len(X_opt)
            reproj1 = np.zeros((n_pts,2)); reproj2 = np.zeros((n_pts,2))
            P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
            P2 = K @ np.hstack([R_opt, t_opt])
            for i in range(n_pts):
                Xh = np.append(X_opt[i],1.0)
                p1 = P1 @ Xh; reproj1[i] = p1[:2]/(p1[2]+1e-15)
                p2 = P2 @ Xh; reproj2[i] = p2[:2]/(p2[2]+1e-15)

            img1_vis = img1_corr.copy(); img2_vis = img2_corr.copy()
            for i in range(n_pts):
                c = (0,255,0)
                cv2.circle(img1_vis, (int(pts1_inliers[i,0]), int(pts1_inliers[i,1])), 6, c, 2)
                cv2.circle(img1_vis, (int(reproj1[i,0]), int(reproj1[i,1])), 4, (255,0,0), 2)
                cv2.circle(img2_vis, (int(pts2_inliers[i,0]), int(pts2_inliers[i,1])), 6, c, 2)
                cv2.circle(img2_vis, (int(reproj2[i,0]), int(reproj2[i,1])), 4, (255,0,0), 2)

            cv2.imwrite(os.path.join(outdir, 'reproj_overlay_ba_img1.png'), img1_vis)
            cv2.imwrite(os.path.join(outdir, 'reproj_overlay_ba_img2.png'), img2_vis)
            config.info('[INFO] BA overlays saved in output/')

            # Replace with optimized and update reported reprojection errors
            R = R_opt; t = t_opt; puntos_3d = X_opt
            err_cam1 = float(err1_ba)
            err_cam2 = float(err2_ba)
        except Exception as e:
            config.info(f"[WARN] Bundle adjustment failed: {e}")

    # --- Diagnóstico adicional: reproyección por punto y overlays ---
    # Construir P1, P2
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    n_pts = len(puntos_3d)
    reproj1 = np.zeros((n_pts, 2))
    reproj2 = np.zeros((n_pts, 2))
    err1_arr = np.zeros(n_pts)
    err2_arr = np.zeros(n_pts)

    for i in range(n_pts):
        X_h = np.append(puntos_3d[i], 1.0)
        p1_proj = P1 @ X_h
        p1_proj = p1_proj[:2] / (p1_proj[2] + 1e-15)
        reproj1[i] = p1_proj
        err1_arr[i] = np.linalg.norm(p1_proj - pts1_inliers[i])

        p2_proj = P2 @ X_h
        p2_proj = p2_proj[:2] / (p2_proj[2] + 1e-15)
        reproj2[i] = p2_proj
        err2_arr[i] = np.linalg.norm(p2_proj - pts2_inliers[i])

    # Guardar CSV con errores por punto
    if not config.SHOW_GUI:
        import csv
        csv_path = os.path.join(outdir, 'reprojection_errors.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['idx', 'u1', 'v1', 'u2', 'v2', 'reproj_u1', 'reproj_v1', 'reproj_u2', 'reproj_v2', 'err1_px', 'err2_px'])
            for i in range(n_pts):
                writer.writerow([
                    i,
                    float(pts1_inliers[i,0]), float(pts1_inliers[i,1]),
                    float(pts2_inliers[i,0]), float(pts2_inliers[i,1]),
                    float(reproj1[i,0]), float(reproj1[i,1]),
                    float(reproj2[i,0]), float(reproj2[i,1]),
                    float(err1_arr[i]), float(err2_arr[i])
                ])
        config.info(f"[INFO] Reprojection errors saved: {csv_path}")

        # Crear overlays visuales con reproyecciones
        img1_vis = img1_corr.copy()
        img2_vis = img2_corr.copy()
        for i in range(n_pts):
            c = (0, 255, 0) if max(err1_arr[i], err2_arr[i]) < 5.0 else (0, 0, 255)
            cv2.circle(img1_vis, (int(pts1_inliers[i,0]), int(pts1_inliers[i,1])), 8, c, 2)
            cv2.circle(img1_vis, (int(reproj1[i,0]), int(reproj1[i,1])), 6, (255, 0, 0), 2)
            cv2.putText(img1_vis, str(i), (int(pts1_inliers[i,0])+5, int(pts1_inliers[i,1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

            cv2.circle(img2_vis, (int(pts2_inliers[i,0]), int(pts2_inliers[i,1])), 8, c, 2)
            cv2.circle(img2_vis, (int(reproj2[i,0]), int(reproj2[i,1])), 6, (255, 0, 0), 2)
            cv2.putText(img2_vis, str(i), (int(pts2_inliers[i,0])+5, int(pts2_inliers[i,1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

        cv2.imwrite(os.path.join(outdir, 'reproj_overlay_img1.png'), img1_vis)
        cv2.imwrite(os.path.join(outdir, 'reproj_overlay_img2.png'), img2_vis)
        config.info(f"[INFO] Reprojection overlays saved in: {outdir}")

    # ===================================================================
    # PASO 8: Visualización 3D
    # ===================================================================
    separador("PASO 8: VISUALIZACIÓN 3D")

    # Visualización 3D: si SHOW_GUI es False, la función de visualización
    # (matplotlib) seguirá generando una figura; guardamos en output/ si es
    # necesario.
    if config.SHOW_GUI:
        visualizar_reconstruccion(puntos_3d, R, t, titulo="Reconstrucción 3D — Par Estéreo")
    else:
        visualizar_reconstruccion(puntos_3d, R, t, titulo=os.path.join(outdir, 'reconstruccion_3d.png'))

    # ===================================================================
    # Resumen final
    # ===================================================================
    separador("RESUMEN DEL PIPELINE")
    config.info(f"""
Calibración:
    - Error de reproyección (calibración):   {error_calib:.4f} px

Geometría Epipolar:
    - Correspondencias ArUco:                {len(pts1)} puntos
    - Inliers tras RANSAC:                   {np.sum(mascara)} puntos
    - Error epipolar medio:                  {error_epi_medio:.6e}

Reconstrucción 3D:
    - Pose seleccionada:                     {idx_pose + 1} de 4
    - Puntos 3D reconstruidos:               {len(puntos_3d)}
    - Error reproyección (cám 1):            {err_cam1:.4f} px
    - Error reproyección (cám 2):            {err_cam2:.4f} px
""")

    # Guardar resultados clave en JSON para la memoria
    results = {
        'K': K.tolist(),
        'error_calibracion_px': float(error_calib),
        'num_correspondencias': int(len(pts1)),
        'num_inliers': int(np.sum(mascara)),
        'error_epipolar_medio': float(error_epi_medio),
        'F': F.tolist(),
        'E': E.tolist(),
        'pose_idx': int(idx_pose),
        'R': R.tolist(),
        't': t.ravel().tolist(),
        'puntos_3d_count': int(len(puntos_3d)),
        'error_reproyeccion_cam1_px': float(err_cam1),
        'error_reproyeccion_cam2_px': float(err_cam2),
    }

    if not config.SHOW_GUI:
        results_path = os.path.join(outdir, 'results.json')
        config.save_json(results, results_path)
        config.info(f"[INFO] Resultados guardados en: {results_path}")

    config.info("[OK] Pipeline completado con éxito.\n")

    # --- Etapa adicional: rectificación estéreo y disparidad densa ---
    try:
        config.separator("ETAPA ADICIONAL: RECTIFICACIÓN Y DISPARIDAD DENSA")

        # stereoRectify (asumimos K y dist son los de ambas cámaras)
        # imageSize debe ser (width, height)
        R1, R2, P1r, P2r, Q, roi1, roi2 = cv2.stereoRectify(
            K, dist, K, dist, img_size, R, t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1r, img_size, cv2.CV_16SC2)
        map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2r, img_size, cv2.CV_16SC2)

        # Use original images for rectification remap (avoid double-undistort)
        rect1 = cv2.remap(img1_orig, map1x, map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2_orig, map2x, map2y, cv2.INTER_LINEAR)

        # Guardar rectificadas
        rect1_path = os.path.join(outdir, 'rectified_left.png')
        rect2_path = os.path.join(outdir, 'rectified_right.png')
        cv2.imwrite(rect1_path, rect1)
        cv2.imwrite(rect2_path, rect2)
        config.info(f"[INFO] Rectified images saved: {rect1_path}, {rect2_path}")

        # Transformar esquinas inliers a coordenadas rectificadas
        if len(pts1_inliers) > 0:
            pts1_und = cv2.undistortPoints(pts1_inliers.reshape(-1,1,2), K, dist, R=R1, P=P1r)
            pts2_und = cv2.undistortPoints(pts2_inliers.reshape(-1,1,2), K, dist, R=R2, P=P2r)
            pts1_rect = pts1_und.reshape(-1,2)
            pts2_rect = pts2_und.reshape(-1,2)

            # Error vertical antes y después
            vert_err_before = np.mean(np.abs(pts1_inliers[:,1] - pts2_inliers[:,1]))
            vert_err_after = np.mean(np.abs(pts1_rect[:,1] - pts2_rect[:,1]))
            config.info(f"[INFO] Error vertical medio antes: {vert_err_before:.3f} px, después: {vert_err_after:.3f} px")
        else:
            vert_err_before = None
            vert_err_after = None

        # Disparity using StereoSGBM
        gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)

        window_size = 5
        min_disp = 0
        num_disp = 16*8  # must be divisible by 16
        matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8*3*window_size**2,
            P2=32*3*window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        disp = matcher.compute(gray1, gray2).astype(np.float32) / 16.0
        # Normalize for visualization
        disp_vis = (disp - min_disp) / num_disp
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_color = (255 * disp_vis).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_color, cv2.COLORMAP_JET)

        disp_path = os.path.join(outdir, 'disparity.png')
        cv2.imwrite(disp_path, disp_color)
        config.info(f"[INFO] Disparity map saved: {disp_path}")

        # Add to results and save updated JSON
        results['vertical_error_before_px'] = float(vert_err_before) if vert_err_before is not None else None
        results['vertical_error_after_px'] = float(vert_err_after) if vert_err_after is not None else None
        results['disparity_path'] = disp_path
        results['rectified_left'] = rect1_path
        results['rectified_right'] = rect2_path

        results_path = os.path.join(outdir, 'results.json')
        config.save_json(results, results_path)
        config.info(f"[INFO] Results updated with rectification/disparity: {results_path}")

    except Exception as e:
        config.info(f"[WARN] Rectification/disparity stage failed: {e}")


if __name__ == "__main__":
    main()
