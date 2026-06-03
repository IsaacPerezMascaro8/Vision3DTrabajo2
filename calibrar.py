"""
calibrar.py
===========
Script independiente de calibración de cámara.

Ejecuta la calibración con las imágenes del directorio dataset/calibracion
y muestra por terminal la matriz K y los coeficientes de distorsión para
copiarlos al main.py.

Uso:
    python3 calibrar.py
"""

import os
import sys
import numpy as np
import cv2
from calibracion import calibrar_camara

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DIR_CALIBRACION = "FotosCalibracion"
DIR_PAR_ESTEREO = "FotosOriginales"
IMG1_PATH       = os.path.join(DIR_PAR_ESTEREO, "1.png")


def main():
    np.set_printoptions(precision=10, suppress=True)

    # Leer resolución del par estéreo para filtrar imágenes de calibración
    img_ref = cv2.imread(IMG1_PATH)
    if img_ref is None:
        print(f"[ERROR] No se pudo leer: {IMG1_PATH}")
        sys.exit(1)
    target_size = (img_ref.shape[1], img_ref.shape[0])

    print(f"Resolución objetivo (par estéreo): {target_size[0]}x{target_size[1]}")
    print("Calibrando...\n")

    K, dist, rvecs, tvecs, error, img_size = calibrar_camara(
        DIR_CALIBRACION,
        mostrar=False,
        target_size=target_size,
    )

    print("=" * 70)
    print("  RESULTADO DE LA CALIBRACIÓN")
    print("=" * 70)
    print(f"\nError de reproyección: {error:.4f} px")
    print(f"Tamaño de imagen:     {img_size}")
    print(f"\nMatriz intrínseca K:")
    print(f"  fx = {K[0,0]:.6f}")
    print(f"  fy = {K[1,1]:.6f}")
    print(f"  cx = {K[0,2]:.6f}")
    print(f"  cy = {K[1,2]:.6f}")
    print(f"\nCoeficientes de distorsión:")
    print(f"  {dist.ravel()}")

    # Imprimir en formato copy-paste para main.py
    print("\n" + "=" * 70)
    print("  COPIAR EN main.py:")
    print("=" * 70)
    print(f"""
K = np.array([
    [{K[0,0]:.10f}, {K[0,1]:.10f}, {K[0,2]:.10f}],
    [{K[1,0]:.10f}, {K[1,1]:.10f}, {K[1,2]:.10f}],
    [{K[2,0]:.10f}, {K[2,1]:.10f}, {K[2,2]:.10f}],
])

dist = np.array([{', '.join(f'{d:.10f}' for d in dist.ravel())}])
""")


if __name__ == "__main__":
    main()
