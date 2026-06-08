import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def calculate_ssd(block_left, block_right):
    diff = block_left.astype(np.float32) - block_right.astype(np.float32)
    return np.sum(diff ** 2)

def main():
    out_dir = "output"
    left_path = os.path.join(out_dir, "rectified_left.png")
    right_path = os.path.join(out_dir, "rectified_right.png")

    if not os.path.isfile(left_path) or not os.path.isfile(right_path):
        print(f"[ERROR] No se encontraron las imágenes en {out_dir}")
        sys.exit(1)

    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    # Reducimos a escala 0.5 igual que en disparidad_densa_ssd.py para coherencia
    escala = 0.5
    h_orig, w_orig = img_left.shape
    new_h, new_w = int(h_orig * escala), int(w_orig * escala)
    img_left = cv2.resize(img_left, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_right = cv2.resize(img_right, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Parámetros de coordenadas
    # NOTA: Ajusta Y_FIXED, X_ARUCO y X_WALL según dónde caigan exactamente
    # el marcador y la pared en la imagen después de reescalar a la mitad.
    Y_FIXED = new_h // 2        # Línea de escaneo epipolar (ej. mitad de la imagen)
    X_ARUCO = new_w // 2        # Punto con alta textura (ej. sobre un marcador ArUco)
    X_WALL = new_w // 4         # Punto sin textura (ej. sobre la pared izquierda lisa)
    
    WIN_SIZE = 9
    MAX_DISP = 64
    margen = WIN_SIZE // 2

    if Y_FIXED < margen or Y_FIXED >= new_h - margen:
        print("[ERROR] Y_FIXED está muy cerca del borde.")
        return
    
    print(f"Analizando perfiles SSD en línea Y={Y_FIXED}")
    print(f"Punto ArUco (Textura): X={X_ARUCO}")
    print(f"Punto Pared (Liso): X={X_WALL}")

    # Extraemos bloques (9x9) en la imagen izquierda
    block_aruco_left = img_left[Y_FIXED-margen : Y_FIXED+margen+1, X_ARUCO-margen : X_ARUCO+margen+1]
    block_wall_left  = img_left[Y_FIXED-margen : Y_FIXED+margen+1, X_WALL-margen : X_WALL+margen+1]

    errores_aruco = []
    errores_wall = []
    disparidades = list(range(MAX_DISP))

    for d in disparidades:
        # Coste para el ArUco
        x_r_aruco = X_ARUCO - d
        if margen <= x_r_aruco < new_w - margen:
            block_aruco_right = img_right[Y_FIXED-margen : Y_FIXED+margen+1, x_r_aruco-margen : x_r_aruco+margen+1]
            ssd_aruco = calculate_ssd(block_aruco_left, block_aruco_right)
            errores_aruco.append(ssd_aruco)
        else:
            errores_aruco.append(np.nan)

        # Coste para la Pared
        x_r_wall = X_WALL - d
        if margen <= x_r_wall < new_w - margen:
            block_wall_right = img_right[Y_FIXED-margen : Y_FIXED+margen+1, x_r_wall-margen : x_r_wall+margen+1]
            ssd_wall = calculate_ssd(block_wall_left, block_wall_right)
            errores_wall.append(ssd_wall)
        else:
            errores_wall.append(np.nan)

    # Graficamos la comparativa
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(disparidades, errores_aruco, marker='o', color='blue', label='SSD ArUco (Textura)')
    plt.title('Perfil SSD - Zona Texturizada (ArUco)')
    plt.xlabel('Disparidad $d$')
    plt.ylabel('Error SSD')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(disparidades, errores_wall, marker='o', color='red', label='SSD Pared (Lisa)')
    plt.title('Perfil SSD - Zona Lisa (Pared)')
    plt.xlabel('Disparidad $d$')
    plt.ylabel('Error SSD')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    out_plot = os.path.join(out_dir, "perfil_ssd_comparativa.png")
    plt.savefig(out_plot)
    print(f"\nGráfica comparativa guardada exitosamente en '{out_plot}'")
    plt.show()

if __name__ == "__main__":
    main()
