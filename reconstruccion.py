"""
reconstruccion.py
=================
Módulo de reconstrucción 3D:
  - Descomposición de la Matriz Esencial (E) en 4 posibles poses (R, t).
  - Triangulación lineal de puntos 3D.
  - Verificación de quiralidad (Z > 0 en ambas cámaras).
  - Selección automática de la pose correcta.
  - Visualización 3D de la nube de puntos reconstruida.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import config
import os
from scipy.optimize import least_squares



# ---------------------------------------------------------------------------
# 1. Descomposición de E en 4 poses candidatas
# ---------------------------------------------------------------------------

def descomponer_esencial(E):
    """
    Descompone la Matriz Esencial E en las 4 poses posibles (R, t).

    Según la factorización de Hartley & Zisserman:
        E = U · diag(1,1,0) · V^T
        W = [[0,-1,0],[1,0,0],[0,0,1]]

    Las 4 soluciones son:
        (R1, t),  (R1, -t),  (R2, t),  (R2, -t)
    donde:
        R1 = U · W   · V^T
        R2 = U · W^T · V^T
        t  = ±U[:, 2]

    Parámetros
    ----------
    E : np.ndarray (3x3)
        Matriz Esencial.

    Retorna
    -------
    poses : list[tuple(np.ndarray, np.ndarray)]
        Lista de 4 tuplas (R, t) con R (3x3) y t (3x1).
    """
    U, S, Vt = np.linalg.svd(E)

    # Asegurar que det(U) > 0 y det(Vt) > 0
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    t = U[:, 2].reshape(3, 1)
    t = t / (np.linalg.norm(t) + 1e-15)   # normalizar

    poses = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t),
    ]

    config.info("[INFO] Descomposición de E → 4 poses candidatas generadas.")
    for i, (R, tvec) in enumerate(poses):
        config.info(f"  Pose {i+1}: det(R) = {np.linalg.det(R):.4f}, "
                    f"t = [{tvec[0,0]:.4f}, {tvec[1,0]:.4f}, {tvec[2,0]:.4f}]")

    return poses


# ---------------------------------------------------------------------------
# 2. Triangulación lineal (DLT)
# ---------------------------------------------------------------------------

def triangular_punto(P1, P2, p1, p2):
    """
    Triangulación lineal de un único punto 3D usando el método DLT.

    Dado un par de matrices de proyección P1, P2 y los puntos
    correspondientes p1 = (x1, y1), p2 = (x2, y2), resuelve el
    sistema A · X = 0 por SVD.

    Parámetros
    ----------
    P1 : np.ndarray (3x4)
        Matriz de proyección de la cámara 1.
    P2 : np.ndarray (3x4)
        Matriz de proyección de la cámara 2.
    p1 : array-like (2,)
        Punto 2D en la imagen 1.
    p2 : array-like (2,)
        Punto 2D en la imagen 2.

    Retorna
    -------
    X : np.ndarray (3,)
        Coordenadas 3D del punto triangulado.
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    A = np.array([
        x1 * P1[2, :] - P1[0, :],
        y1 * P1[2, :] - P1[1, :],
        x2 * P2[2, :] - P2[0, :],
        y2 * P2[2, :] - P2[1, :],
    ], dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1, :]
    X = X_hom[:3] / (X_hom[3] + 1e-15)

    return X


def triangular_puntos(P1, P2, pts1, pts2):
    """
    Triangula múltiples correspondencias.

    Parámetros
    ----------
    P1, P2 : np.ndarray (3x4)
        Matrices de proyección.
    pts1, pts2 : np.ndarray (Nx2)
        Correspondencias de puntos.

    Retorna
    -------
    puntos_3d : np.ndarray (Nx3)
        Nube de puntos 3D.
    """
    n = len(pts1)
    puntos_3d = np.zeros((n, 3))

    for i in range(n):
        puntos_3d[i] = triangular_punto(P1, P2, pts1[i], pts2[i])

    return puntos_3d


# ---------------------------------------------------------------------------
# 3. Verificación de quiralidad y selección de pose
# ---------------------------------------------------------------------------

def verificar_quiralidad(R, t, K, pts1, pts2):
    """
    Triangula puntos y cuenta cuántos tienen Z > 0 en AMBAS cámaras.

    Parámetros
    ----------
    R : np.ndarray (3x3)
        Rotación de la cámara 2 respecto a la 1.
    t : np.ndarray (3x1)
        Translación de la cámara 2 respecto a la 1.
    K : np.ndarray (3x3)
        Matriz intrínseca.
    pts1, pts2 : np.ndarray (Nx2)
        Correspondencias.

    Retorna
    -------
    num_delante : int
        Número de puntos con Z > 0 en ambas cámaras.
    puntos_3d : np.ndarray (Nx3)
        Nube de puntos 3D triangulados.
    """
    # Cámara 1: P1 = K · [I | 0]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    # Cámara 2: P2 = K · [R | t]
    P2 = K @ np.hstack([R, t])

    puntos_3d = triangular_puntos(P1, P2, pts1, pts2)

    # Verificar Z > 0 en cámara 1 (coordenadas mundo = cámara 1)
    z_cam1 = puntos_3d[:, 2]

    # Verificar Z > 0 en cámara 2
    puntos_cam2 = (R @ puntos_3d.T + t).T
    z_cam2 = puntos_cam2[:, 2]

    delante = (z_cam1 > 0) & (z_cam2 > 0)
    num_delante = np.sum(delante)

    return num_delante, puntos_3d


def seleccionar_pose(E, K, pts1, pts2):
    """
    Descompone E en 4 poses, triangula para cada una y selecciona
    la pose con mayor número de puntos con Z > 0 en ambas cámaras
    (verificación de quiralidad).

    Parámetros
    ----------
    E : np.ndarray (3x3)
        Matriz Esencial.
    K : np.ndarray (3x3)
        Matriz intrínseca.
    pts1, pts2 : np.ndarray (Nx2)
        Correspondencias de puntos.

    Retorna
    -------
    R_mejor : np.ndarray (3x3)
        Rotación de la mejor pose.
    t_mejor : np.ndarray (3x1)
        Translación de la mejor pose.
    puntos_3d : np.ndarray (Nx3)
        Nube de puntos 3D de la mejor pose.
    idx_mejor : int
        Índice (0-3) de la pose seleccionada.
    """
    poses = descomponer_esencial(E)

    mejor_num = -1
    R_mejor = None
    t_mejor = None
    puntos_3d_mejor = None
    idx_mejor = -1

    config.info("\n[INFO] Verificación de quiralidad (Z > 0 en ambas cámaras):")
    for i, (R, t) in enumerate(poses):
        num, pts_3d = verificar_quiralidad(R, t, K, pts1, pts2)
        config.info(f"  Pose {i+1}: {num}/{len(pts1)} puntos delante de ambas cámaras")

        if num > mejor_num:
            mejor_num = num
            R_mejor = R
            t_mejor = t
            puntos_3d_mejor = pts_3d
            idx_mejor = i

    config.info(f"\n[INFO] ✅ Pose seleccionada: {idx_mejor + 1} "
            f"({mejor_num}/{len(pts1)} puntos válidos)")
    config.info(f"  R =\n{R_mejor}")
    config.info(f"  t = {t_mejor.ravel()}")

    return R_mejor, t_mejor, puntos_3d_mejor, idx_mejor


# ---------------------------------------------------------------------------
# 4. Error de reproyección post-triangulación
# ---------------------------------------------------------------------------

def error_reproyeccion(K, R, t, puntos_3d, pts1, pts2):
    """
    Calcula el error de reproyección medio para verificar la calidad
    de la triangulación.

    Retorna
    -------
    error_cam1 : float
        Error medio en la imagen 1 (píxeles).
    error_cam2 : float
        Error medio en la imagen 2 (píxeles).
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    n = len(puntos_3d)
    err1 = np.zeros(n)
    err2 = np.zeros(n)

    for i in range(n):
        X_hom = np.append(puntos_3d[i], 1.0)

        # Reproyección en cámara 1
        proj1 = P1 @ X_hom
        proj1 = proj1[:2] / (proj1[2] + 1e-15)
        err1[i] = np.linalg.norm(proj1 - pts1[i])

        # Reproyección en cámara 2
        proj2 = P2 @ X_hom
        proj2 = proj2[:2] / (proj2[2] + 1e-15)
        err2[i] = np.linalg.norm(proj2 - pts2[i])

    error_cam1 = np.mean(err1)
    error_cam2 = np.mean(err2)

    config.info(f"\n[INFO] Error de reproyección post-triangulación:")
    config.info(f"  Cámara 1: {error_cam1:.4f} px (medio)")
    config.info(f"  Cámara 2: {error_cam2:.4f} px (medio)")

    return error_cam1, error_cam2


# ---------------------------------------------------------------------------
# 5. Visualización 3D
# ---------------------------------------------------------------------------

def visualizar_reconstruccion(puntos_3d, R, t, titulo="Reconstrucción 3D"):
    """
    Visualiza la nube de puntos 3D reconstruida junto con las posiciones
    de las dos cámaras.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Nube de puntos
    ax.scatter(puntos_3d[:, 0], puntos_3d[:, 1], puntos_3d[:, 2],
               c='steelblue', marker='o', s=40, alpha=0.8, label='Puntos 3D')

    # Cámara 1 en el origen
    ax.scatter(0, 0, 0, c='red', marker='^', s=200, label='Cámara 1')

    # Cámara 2
    C2 = (-R.T @ t).ravel()
    ax.scatter(C2[0], C2[1], C2[2], c='green', marker='^', s=200,
               label='Cámara 2')

    # Ejes de la cámara 1
    escala = np.max(np.abs(puntos_3d)) * 0.2
    for eje, color_eje in zip(np.eye(3), ['r', 'g', 'b']):
        ax.quiver(0, 0, 0, eje[0], eje[1], eje[2],
                  length=escala, color=color_eje, linewidth=2)

    # Ejes de la cámara 2
    for eje, color_eje in zip(R.T, ['r', 'g', 'b']):
        ax.quiver(C2[0], C2[1], C2[2], eje[0], eje[1], eje[2],
                  length=escala, color=color_eje, linewidth=2, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(titulo)
    ax.legend(loc='upper left')

    # Ajustar aspecto (robusto ante datos con rango pequeño)
    all_pts = np.vstack([puntos_3d, [[0, 0, 0]], [C2]])
    max_range = max(np.ptp(all_pts, axis=0)) * 0.6
    if max_range < 1e-6:
        max_range = 1.0
    mid = np.mean(all_pts, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    # Si no hay GUI, guardar la figura en output/
    if not config.SHOW_GUI:
        outdir = config.ensure_output_dir('output')
        save_path = titulo if os.path.isabs(titulo) or titulo.endswith('.png') else os.path.join(outdir, f"{titulo.replace(' ', '_')}.png")
        config.info(f"[INFO] Guardando visualización 3D en: {save_path}")
        plt.savefig(save_path)
        plt.close(fig)
    else:
        config.info("[INFO] Mostrando visualización 3D...")
        plt.show()


def _rodrigues_to_matrix(rvec):
    from scipy.spatial.transform import Rotation as R_scipy
    return R_scipy.from_rotvec(rvec.ravel()).as_matrix()


def _matrix_to_rodrigues(R):
    from scipy.spatial.transform import Rotation as R_scipy
    return R_scipy.from_matrix(R).as_rotvec()


def bundle_adjustment(K, r_init, t_init, X_init, pts1, pts2, max_nfev=200, two_phase=True):
    """
    Refinamiento por mínimos cuadrados no-lineal de la pose (rvec,t)
    y de los puntos 3D X_init minimizando el error de reproyección.

    Parámetros
    ---------
    K : (3,3) intrínsecas
    r_init : (3,) vector Rodrigues inicial
    t_init : (3,) translación inicial
    X_init : (N,3) puntos 3D iniciales
    pts1, pts2 : (N,2) correspondencias en ambas imágenes (deben corresponder a X_init)

    Retorna
    -------
    r_opt, t_opt, X_opt
    """
def pack(rvec, tvec, X):
    return np.hstack([rvec, tvec, X.ravel()])

def unpack(x):
    rvec = x[0:3]
    tvec = x[3:6]
    X = x[6:].reshape((-1, 3))
    return rvec, tvec, X

def reproj_errors(x, K, pts1, pts2):
    rvec, tvec, X = unpack(x)
    R = _rodrigues_to_matrix(rvec)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, tvec.reshape(3,1)])

    errs = []
    for i in range(len(X)):
        Xh = np.append(X[i], 1.0)
        p1 = P1 @ Xh
        p1 = p1[:2] / (p1[2] + 1e-15)
        p2 = P2 @ Xh
        p2 = p2[:2] / (p2[2] + 1e-15)
        errs.append(p1 - pts1[i])
        errs.append(p2 - pts2[i])
    return np.hstack(errs)

def reproj_pose_only(x, K, X0, pts1, pts2):
    rvec = x[0:3]
    tvec = x[3:6]
    R = _rodrigues_to_matrix(rvec)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, tvec.reshape(3,1)])
    errs = []
    for i in range(len(X0)):
        Xh = np.append(X0[i], 1.0)
        p1 = P1 @ Xh; p1 = p1[:2] / (p1[2] + 1e-15)
        p2 = P2 @ Xh; p2 = p2[:2] / (p2[2] + 1e-15)
        errs.append(p1 - pts1[i]); errs.append(p2 - pts2[i])
    return np.hstack(errs)

def bundle_adjustment(K, r_init, t_init, X_init, pts1, pts2, max_nfev=200, two_phase=True):
    n = len(X_init)
    rvec0 = r_init.copy()
    t0 = t_init.ravel().copy()
    X0 = X_init.copy()

    if two_phase:
        x0_pose = np.hstack([rvec0, t0])
        res1 = least_squares(reproj_pose_only, x0_pose, args=(K, X0, pts1, pts2), verbose=0, max_nfev=max_nfev//4, ftol=1e-8, xtol=1e-8, loss='huber')
        rvec_opt = res1.x[0:3]
        t_opt = res1.x[3:6]

        x0 = pack(rvec_opt, t_opt, X0)
        res2 = least_squares(reproj_errors, x0, args=(K, pts1, pts2), verbose=2, max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, loss='huber')
        rvec_opt, t_opt, X_opt = unpack(res2.x)
        return rvec_opt, t_opt.reshape(3,1), X_opt
    else:
        x0 = pack(r_init, t_init.ravel(), X_init)
        res = least_squares(reproj_errors, x0, args=(K, pts1, pts2), verbose=2, max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, loss='huber')
        rvec_opt, t_opt, X_opt = unpack(res.x)
        return rvec_opt, t_opt.reshape(3,1), X_opt


# ---------------------------------------------------------------------------
# Ejecución directa (para pruebas)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config.info("Este módulo se usa desde main.py o importándolo directamente.")
    config.info("Funciones disponibles:")
    config.info("  - descomponer_esencial(E)")
    config.info("  - triangular_puntos(P1, P2, pts1, pts2)")
    config.info("  - seleccionar_pose(E, K, pts1, pts2)")
    config.info("  - error_reproyeccion(K, R, t, puntos_3d, pts1, pts2)")
    config.info("  - visualizar_reconstruccion(puntos_3d, R, t)")
