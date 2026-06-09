import cv2
import numpy as np
import os
import config

def rectificar_par_estereo_no_calibrado(img1, img2, F, pts1, pts2):
    """
    Rectificación usando el algoritmo de Hartley (no calibrado).
    Calcula las homografías directamente de la Matriz Fundamental y los puntos,
    absorbiendo cualquier diferencia de distancia focal o centro óptico entre las fotos.
    """
    img_size = (img1.shape[1], img1.shape[0])
    
    p1 = pts1.astype(np.float32).reshape(-1, 2)
    p2 = pts2.astype(np.float32).reshape(-1, 2)
    
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(p1, p2, F, img_size)
    
    img_rect1 = cv2.warpPerspective(img1, H1, img_size)
    img_rect2 = cv2.warpPerspective(img2, H2, img_size)
    
    return img_rect1, img_rect2, H1, H2

def dibujar_lineas_horizontales(img_rect1, img_rect2, pts1=None, pts2=None, H1=None, H2=None, num_lineas=20):
    """
    Dibuja líneas horizontales paralelas sobre ambas imágenes rectificadas
    para demostrar la correcta rectificación epipolar.
    Si se proporcionan los puntos originales, los mapea al plano rectificado
    y dibuja las líneas pasando exactamente por ellos.
    """
    img_concat = np.hstack([img_rect1, img_rect2])
    h, w = img_concat.shape[:2]
    w_half = img_rect1.shape[1]
    
    if pts1 is not None and pts2 is not None and H1 is not None and H2 is not None:
        pts1_rect = cv2.perspectiveTransform(pts1.astype(np.float32).reshape(-1, 1, 2), H1).reshape(-1, 2)
        pts2_rect = cv2.perspectiveTransform(pts2.astype(np.float32).reshape(-1, 1, 2), H2).reshape(-1, 2)
        
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

def afinar_rectificacion(R, angulo_roll_grados):
    """
    Aplica una corrección fina de 'roll' (rotación sobre el eje Z) 
    para enderezar las líneas epipolares en los bordes.
    """
    rad = np.radians(angulo_roll_grados)
    c, s = np.cos(rad), np.sin(rad)
    R_correccion = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R_correccion @ R
