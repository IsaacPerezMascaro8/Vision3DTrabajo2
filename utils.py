import os
import csv
import cv2
import numpy as np
import config

def mostrar_img(titulo, img, max_h=800, save_path=None, outdir='output'):
    if config.SHOW_GUI:
        ratio = max_h / img.shape[0]
        img_show = cv2.resize(img, None, fx=ratio, fy=ratio)
        cv2.imshow(titulo, img_show)
    else:
        outdir = config.ensure_output_dir(outdir)
        if save_path is None:
            save_path = os.path.join(outdir, f"{titulo.replace(' ', '_')}.png")
        cv2.imwrite(save_path, img)
        config.info(f"  → Imagen guardada: {save_path}")

def guardar_errores_csv(outdir, n_pts, pts1_inliers, pts2_inliers, reproj1, reproj2, err1_arr, err2_arr):
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

def dibujar_overlays_reproyeccion(img1, img2, outdir, n_pts, pts1_inliers, pts2_inliers, reproj1, reproj2, err1_arr, err2_arr):
    img1_vis = img1.copy()
    img2_vis = img2.copy()
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

def guardar_resultados_json(results, outdir, filename='results.json'):
    results_path = os.path.join(outdir, filename)
    config.save_json(results, results_path)
    config.info(f"[INFO] Resultados guardados en: {results_path}")
