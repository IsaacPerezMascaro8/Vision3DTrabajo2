import cv2
import numpy as np
import os
import config

def rectificar_par_estereo(img1, img2, K, R, t):
    """
    Rectificación matemática de imágenes estereoscópicas.
    """
    img_size = (img1.shape[1], img1.shape[0])
    dist_null = np.zeros(5)

    R1, R2, P1r, P2r, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist_null, K, dist_null, img_size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)

    map1x, map1y = cv2.initUndistortRectifyMap(K, dist_null, R1, P1r, img_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist_null, R2, P2r, img_size, cv2.CV_16SC2)

    img_rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    return img_rect1, img_rect2, R1, P1r, R2, P2r

def dibujar_lineas_horizontales(img_rect1, img_rect2, pts1=None, pts2=None, K=None, R1=None, P1=None, R2=None, P2=None, num_lineas=20):
    """
    Dibuja líneas horizontales paralelas sobre ambas imágenes rectificadas
    para demostrar la correcta rectificación epipolar.
    Si se proporcionan los puntos originales, los mapea al plano rectificado
    y dibuja las líneas pasando exactamente por ellos.
    """
    img_concat = np.hstack([img_rect1, img_rect2])
    h, w = img_concat.shape[:2]
    w_half = img_rect1.shape[1]
    
    if pts1 is not None and pts2 is not None and K is not None:
        dist_nula = np.zeros(5)
        # Transformar los puntos 2D originales al nuevo plano rectificado
        pts1_rect = cv2.undistortPoints(pts1.astype(np.float32).reshape(-1, 1, 2), K, dist_nula, R=R1, P=P1)
        pts2_rect = cv2.undistortPoints(pts2.astype(np.float32).reshape(-1, 1, 2), K, dist_nula, R=R2, P=P2)
        
        pts1_rect = pts1_rect.reshape(-1, 2)
        pts2_rect = pts2_rect.reshape(-1, 2)
        
        for pt1, pt2 in zip(pts1_rect, pts2_rect):
            y = int(pt1[1])
            color = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.circle(img_concat, (int(pt1[0]), y), 8, color, -1)
            cv2.circle(img_concat, (int(pt2[0]) + w_half, int(pt2[1])), 8, color, -1)
            cv2.line(img_concat, (0, y), (w, y), color, 2)
    else:
        paso = max(1, h // num_lineas)
        for y in range(paso, h, paso):
            color = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.line(img_concat, (0, y), (w, y), color, 2)
            
    return img_concat
