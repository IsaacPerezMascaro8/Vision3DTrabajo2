import os
import sys
import numpy as np
import cv2
import config
from cli import parse_arguments
from utils import mostrar_img, guardar_errores_csv, dibujar_overlays_reproyeccion, guardar_resultados_json
from rectificacion import rectificar_par_estereo, dibujar_lineas_horizontales
from geometria_epipolar import obtener_correspondencias_aruco, ransac_fundamental, calcular_esencial
from reconstruccion import seleccionar_pose, error_reproyeccion, visualizar_reconstruccion, triangular_punto

K = np.array([[3954.809289, 0.0, 2133.663739], [0.0, 3956.467881, 2889.190270], [0.0, 0.0, 1.0]])

def main():
    args, outdir = parse_arguments()
    
    # 1. Cargar imágenes
    img1 = cv2.imread(os.path.join("FotosOriginales", "1.png"))
    img2 = cv2.imread(os.path.join("FotosOriginales", "2.png"))
    if img1 is None or img2 is None:
        sys.exit("Error: No se pudieron cargar las imágenes.")

    # 2. Detección ArUco
    pts1, pts2, ids, p_ids = obtener_correspondencias_aruco(img1, img2, refine_corners=args.refine_corners)
    
    # 3. Matriz Fundamental y Esencial
    F, mask = ransac_fundamental(pts1, pts2, umbral=args.ransac_threshold, max_iter=5000, marker_ids=p_ids)
    p1_in, p2_in = pts1[mask], pts2[mask]
    E = calcular_esencial(F, K)
    
    # 4. Triangulación
    R, t, X, _ = seleccionar_pose(E, K, p1_in, p2_in)

    # --- FIJAR ESCALA MÉTRICA ---
    # El lado del ArUco mide 12 cm (0.12 metros)
    TAMANO_ARUCO_M = 0.12

    # Calculamos P1 y P2 temporales con la traslación 't' sin escalar
    P1_temp = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_temp = K @ np.hstack([R, t])

    # Triangulamos las esquinas 0 y 1 del primer ArUco detectado (lado superior)
    X_corner0 = triangular_punto(P1_temp, P2_temp, pts1[0], pts2[0])
    X_corner1 = triangular_punto(P1_temp, P2_temp, pts1[1], pts2[1])

    # Distancia 3D virtual y factor de escala
    distancia_3d = np.linalg.norm(X_corner0 - X_corner1)
    factor_escala = TAMANO_ARUCO_M / distancia_3d
    config.info(f"[INFO] Factor de escala calculado: {factor_escala:.4f}")

    # Aplicar la escala al vector de traslación y a toda la nube de puntos
    t = t * factor_escala
    X = X * factor_escala
    # ---------------------------

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    reproj1, reproj2 = np.zeros((len(X), 2)), np.zeros((len(X), 2))
    err1_arr, err2_arr = np.zeros(len(X)), np.zeros(len(X))
    for i in range(len(X)):
        Xh = np.append(X[i], 1.0)
        p1 = P1 @ Xh; p1 = p1[:2] / (p1[2] + 1e-15); reproj1[i] = p1; err1_arr[i] = np.linalg.norm(p1 - p1_in[i])
        p2 = P2 @ Xh; p2 = p2[:2] / (p2[2] + 1e-15); reproj2[i] = p2; err2_arr[i] = np.linalg.norm(p2 - p2_in[i])

    # Guardar resultados analíticos
    if not config.SHOW_GUI:
        guardar_errores_csv(outdir, len(X), p1_in, p2_in, reproj1, reproj2, err1_arr, err2_arr)
        dibujar_overlays_reproyeccion(img1, img2, outdir, len(X), p1_in, p2_in, reproj1, reproj2, err1_arr, err2_arr)
        guardar_resultados_json({'K': K.tolist(), 'err1': np.mean(err1_arr), 'err2': np.mean(err2_arr)}, outdir)

    # 5. Rectificación y dibujo de líneas horizontales
    img_rect1, img_rect2, R1, P1r, R2, P2r = rectificar_par_estereo(img1, img2, K, R, t)
    img_rect_lines = dibujar_lineas_horizontales(
        img_rect1, img_rect2, 
        pts1=p1_in, pts2=p2_in, 
        K=K, R1=R1, P1=P1r, R2=R2, P2=P2r
    )
    
    cv2.imwrite(os.path.join(outdir, 'rectified_lines_combined.png'), img_rect_lines)
    print("Pipeline completado. Imagen rectificada guardada en:", os.path.join(outdir, 'rectified_lines_combined.png'))

    if config.SHOW_GUI:
        ratio = 1200 / img_rect_lines.shape[1]
        cv2.imshow("Rectificacion Perfecta", cv2.resize(img_rect_lines, None, fx=ratio, fy=ratio))
        visualizar_reconstruccion(X, R, t, titulo="Reconstrucción 3D")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
