"""
Módulo de correspondencia densa basada en embeddings semánticos DINOv2.

Utiliza el modelo DINOv2 (ViT-B/14) como extractor de características para
calcular un mapa de disparidad semántico, superando las limitaciones del
matching clásico basado en intensidad en zonas sin textura.
"""
import os
import numpy as np
import cv2
import config

import torch
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────
# Paso 1: Carga del modelo DINOv2
# ──────────────────────────────────────────────────────────────────────

def cargar_modelo_dino():
    """Descarga y configura DINOv2 ViT-B/14 desde torch.hub.

    Returns
    -------
    model : torch.nn.Module
        Modelo DINOv2 en modo evaluación.
    device : torch.device
        Dispositivo (cuda si disponible, sino cpu).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.info(f"[DINO] Cargando modelo DINOv2 ViT-B/14 en {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    model.to(device)
    config.info("[DINO] Modelo cargado correctamente.")
    return model, device


# ──────────────────────────────────────────────────────────────────────
# Paso 2: Extracción de embeddings
# ──────────────────────────────────────────────────────────────────────

# Preprocesado estándar de DINOv2 (ImageNet stats)
_DINO_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

PATCH_SIZE = 14  # ViT-B/14


def _ajustar_a_multiplo(img, patch_size=PATCH_SIZE):
    """Redimensiona la imagen para que alto y ancho sean múltiplos del patch."""
    h, w = img.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extraer_embeddings(imagen, model, device):
    """Extrae el mapa de embeddings de una imagen usando DINOv2.

    Parameters
    ----------
    imagen : ndarray (BGR, uint8)
        Imagen de entrada.
    model : torch.nn.Module
        Modelo DINOv2 cargado.
    device : torch.device
        Dispositivo de cómputo.

    Returns
    -------
    feat_map : ndarray de forma (H_patches, W_patches, D)
        Mapa de embeddings espaciales (D = 768 para ViT-B).
    h_patches, w_patches : int
        Dimensiones del mapa en patches.
    """
    # Ajustar a múltiplo de 14
    img_adj = _ajustar_a_multiplo(imagen)
    h, w = img_adj.shape[:2]
    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE

    # BGR → RGB y preprocesado
    img_rgb = cv2.cvtColor(img_adj, cv2.COLOR_BGR2RGB)
    tensor = _DINO_TRANSFORM(img_rgb).unsqueeze(0).to(device)

    # Extraer patch tokens (sin CLS)
    with torch.no_grad():
        features = model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"]  # (1, N_patches, D)

    # Reshape a mapa espacial
    feat_map = patch_tokens.squeeze(0).cpu().numpy()          # (N, D)
    feat_map = feat_map.reshape(h_patches, w_patches, -1)     # (H, W, D)

    return feat_map, h_patches, w_patches


# ──────────────────────────────────────────────────────────────────────
# Paso 3: Matching semántico (SSD sobre embeddings)
# ──────────────────────────────────────────────────────────────────────

def calcular_disparidad_dino(img1, img2, max_disp=128, escala=0.5):
    """Calcula el mapa de disparidad usando correspondencia semántica DINOv2.

    Parameters
    ----------
    img1, img2 : ndarray (BGR)
        Par de imágenes rectificadas.
    max_disp : int
        Ventana máxima de búsqueda de disparidad (en unidades de patches).
    escala : float
        Factor de downsampling previo para reducir el coste computacional.

    Returns
    -------
    disparidad_cruda : ndarray (float32)
        Mapa de disparidad en coordenadas de patches.
    mapa_color : ndarray (uint8, 3 canales)
        Versión coloreada con COLORMAP_MAGMA.
    """
    # --- Downsampling para eficiencia ---
    if escala != 1.0:
        h_orig, w_orig = img1.shape[:2]
        new_h, new_w = int(h_orig * escala), int(w_orig * escala)
        img1_ds = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img2_ds = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        config.info(f"[DINO] Downsampling aplicado: {w_orig}x{h_orig} → {new_w}x{new_h}")
    else:
        img1_ds = img1
        img2_ds = img2

    # --- Cargar modelo ---
    model, device = cargar_modelo_dino()

    # --- Extraer embeddings ---
    config.info("[DINO] Extrayendo embeddings imagen izquierda...")
    feat1, h_p, w_p = extraer_embeddings(img1_ds, model, device)
    config.info("[DINO] Extrayendo embeddings imagen derecha...")
    feat2, _, _ = extraer_embeddings(img2_ds, model, device)

    config.info(f"[DINO] Mapa de features: {h_p}x{w_p}, dim={feat1.shape[2]}")

    # --- Limitar disparidad máxima al ancho disponible ---
    max_d = min(max_disp, w_p)

    # --- Búsqueda de correspondencia SSD sobre embeddings + Ratio Test ---
    RATIO_UMBRAL = 1.2  # Ratio Test de Lowe: min2/min1 debe superar este umbral
    config.info(f"[DINO] Calculando matching semántico (max_disp={max_d}, ratio_umbral={RATIO_UMBRAL})...")
    mapa_dino = np.zeros((h_p, w_p), dtype=np.float32)
    mask_confianza = np.zeros((h_p, w_p), dtype=bool)

    for y in range(h_p):
        for x in range(w_p):
            ref_vec = feat1[y, x]  # vector de embedding de referencia (D,)

            # Rango de búsqueda en la imagen derecha (misma fila epipolar)
            x_min = max(0, x - max_d)
            x_max = x + 1  # la disparidad estéreo es siempre ≥ 0 (izq → der)

            # Extraer todos los candidatos de esa fila de una vez (vectorizado)
            candidatos = feat2[y, x_min:x_max]  # (num_candidatos, D)

            # SSD: distancia euclidiana al cuadrado contra cada candidato
            diff = candidatos - ref_vec  # broadcasting (num_candidatos, D)
            ssd = np.sum(diff * diff, axis=1)  # (num_candidatos,)

            # Necesitamos al menos 2 candidatos para el Ratio Test
            if len(ssd) < 2:
                mapa_dino[y, x] = 0.0
                continue

            # Obtener los dos costes más bajos
            idx_sorted = np.argpartition(ssd, 1)[:2]
            costes_top2 = ssd[idx_sorted]
            min1 = np.min(costes_top2)
            min2 = np.max(costes_top2)
            best_local = idx_sorted[np.argmin(costes_top2)]

            # Ratio Test de Lowe: correspondencia aceptada solo si hay
            # una diferencia clara entre el mejor y el segundo mejor
            if min1 > 0 and (min2 / min1) >= RATIO_UMBRAL:
                best_x = x_min + best_local
                mapa_dino[y, x] = float(x - best_x)
                mask_confianza[y, x] = True
            else:
                mapa_dino[y, x] = 0.0

        # Progreso cada 20% de filas
        if (y + 1) % max(1, h_p // 5) == 0:
            pct = 100 * (y + 1) // h_p
            n_validos = int(np.sum(mask_confianza[:y+1]))
            config.info(f"[DINO]   Progreso: {pct}% — {n_validos} patches válidos")

    config.info("[DINO] Matching semántico completado.")
    n_total = h_p * w_p
    n_conf = int(np.sum(mask_confianza))
    config.info(f"[DINO] Patches de alta confianza: {n_conf}/{n_total} ({100*n_conf/n_total:.1f}%)")

    # --- Post-procesado: limpiar ruido de sal y pimienta ---
    mapa_dino = cv2.medianBlur(mapa_dino.astype(np.float32), 5)

    # --- Aplicar máscara de confianza (zonas ambiguas → 0) ---
    mapa_dino[~mask_confianza] = 0.0

    # --- Filtrado de valores negativos ---
    mapa_dino[mapa_dino < 0] = 0.0

    # --- Normalización para colormap ---
    disp_norm = cv2.normalize(mapa_dino, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)
    mapa_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_MAGMA)

    # Forzar fondo negro puro en zonas de baja confianza
    mapa_color[~mask_confianza] = [0, 0, 0]

    # Liberar memoria GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mapa_dino, mapa_color
