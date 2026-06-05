import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.patches import Rectangle

def demostrar_busqueda_ssd(img1_rect, img2_rect, pt_x, pt_y, win_size=15):
    """Demuestra la búsqueda de correspondencia estereoscópica usando SSD.

    Parameters
    ----------
    img1_rect, img2_rect : ndarray
        Imágenes estéreo ya rectificadas (BGR o RGB).
    pt_x, pt_y : int
        Coordenadas del punto de referencia en la imagen izquierda.
    win_size : int, optional
        Tamaño de la ventana cuadrada (debe ser impar). Por defecto 15.
    """
    # ------------------------------------------------------------------
    # Paso 1: Preparación (grises, float32)
    # ------------------------------------------------------------------
    gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray1.shape
    margen = win_size // 2

    # ------------------------------------------------------------------
    # Paso 2: Extraer plantilla (template) de la izquierda
    # ------------------------------------------------------------------
    if not (margen <= pt_x < w - margen and margen <= pt_y < h - margen):
        raise ValueError("El punto especificado está demasiado cerca del borde para la ventana solicitada.")
    template = gray1[pt_y - margen:pt_y + margen + 1,
                     pt_x - margen:pt_x + margen + 1]

    # ------------------------------------------------------------------
    # Paso 3: Deslizamiento a lo largo de la scanline y cálculo SSD
    # ------------------------------------------------------------------
    ssd_costs = np.full(w, np.nan, dtype=np.float32)
    for x_search in range(margen, w - margen):
        ventana = gray2[pt_y - margen:pt_y + margen + 1,
                       x_search - margen:x_search + margen + 1]
        # SSD = sum((T - V)^2)
        diff = template - ventana
        ssd = np.sum(diff * diff)
        ssd_costs[x_search] = ssd

    # Mejor coincidencia (menor SSD)
    best_x = int(np.nanargmin(ssd_costs))

    # ------------------------------------------------------------------
    # Paso 4: Visualización con Matplotlib (GridSpec)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.1, wspace=0.1)

    # --- Subplot 1 (Arriba Izquierda) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img1_rect, cv2.COLOR_BGR2RGB))
    ax1.set_title('Imagen Izquierda (Referencia)')
    rect = Rectangle((pt_x - margen, pt_y - margen), win_size, win_size,
                     linewidth=2, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.axhline(y=pt_y, color='m', linewidth=1)
    ax1.axis('off')

    # --- Subplot 2 (Arriba Derecha) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(img2_rect, cv2.COLOR_BGR2RGB), aspect='auto')
    ax2.set_title('Imagen Derecha (Búsqueda)')
    ax2.axhline(y=pt_y, color='m', linewidth=1)
    ax2.axvline(x=best_x, color='r', linestyle='--', linewidth=2)
    ax2.set_xlim(0, img2_rect.shape[1])
    ax2.set_ylim(img2_rect.shape[0], 0)
    ax2.axis('off')

    # --- Espacio Abajo Izquierda: Queda vacío ---
    # No añadimos nada en gs[1, 0]

    # --- Subplot 3 (Abajo Derecha - Gráfica SSD) ---
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.plot(ssd_costs, label='SSD')
    ax3.set_xlabel('Coordenada X (pixel)')
    ax3.set_ylabel('Costo SSD')
    ax3.grid(True)
    ax3.axvline(x=best_x, color='r', linestyle='--', linewidth=2, label=f'Mejor X = {best_x}')
    ax3.set_xlim(0, img2_rect.shape[1])
    ax3.margins(x=0)
    ax3.legend()
    if config.SHOW_GUI:
        plt.show()
    else:
        out_dir = config.ensure_output_dir("output")
        out_path = os.path.join(out_dir, "ssd_demonstration.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        config.info(f"[INFO] Demostración SSD guardada en: {out_path}")
    plt.close(fig)
    return best_x, ssd_costs


if __name__ == "__main__":
    # Rutas de imágenes rectificadas (se asume que el pipeline anterior las ha guardado)
    left_path = os.path.join('output', 'rectified_left.png')
    right_path = os.path.join('output', 'rectified_right.png')
    if not os.path.isfile(left_path) or not os.path.isfile(right_path):
        raise FileNotFoundError('Las imágenes rectificadas no se encuentran en la carpeta output/')

    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    # Coordenadas de ejemplo: un punto interior con buen contraste (ajustar según sus imágenes)
    pt_x_demo = 200
    pt_y_demo = 150

    demostrar_busqueda_ssd(img_left, img_right, pt_x_demo, pt_y_demo, win_size=15)
