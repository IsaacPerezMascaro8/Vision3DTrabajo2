# Reconstrucción 3D y Correspondencia Estéreo Densa (Plane Sweeping)

Pipeline completo para la estimación de geometría epipolar, calibración de cámara, reconstrucción 3D métrica usando marcadores ArUco, y generación de mapas de disparidad densa mediante un algoritmo de Plane Sweeping personalizado.

## Arquitectura del Proyecto

**Core y Configuración:**
* `main.py`: Script principal que orquesta todo el pipeline (detección, geometría epipolar, triangulación métrica, rectificación y disparidad).
* `config.py`: Módulo de variables globales para controlar el nivel de verbosidad (logs) y la interfaz gráfica.
* `cli.py`: Analizador de argumentos de línea de comandos (argparse).
* `utils.py`: Funciones auxiliares para guardar imágenes, CSVs de error y datos en JSON.

**Geometría y Reconstrucción 3D:**
* `geometria_epipolar.py`: Detección de ArUcos, normalización de puntos, algoritmo de los 8 puntos y cálculo robusto de la Matriz Fundamental (RANSAC) y Esencial.
* `reconstruccion.py`: Descomposición de la Matriz Esencial, verificación de quiralidad, triangulación DLT, cálculo de error de reproyección y visualización 3D.
* `rectificacion.py`: Rectificación matemática de las imágenes estéreo y dibujo de líneas epipolares horizontales.

**Visión Estéreo Densa:**
* `plane_sweeping.py`: Implementación principal del algoritmo de Plane Sweeping con preprocesado CLAHE, volumen de costes SAD y suavizado mediante Filtro Guiado (Guided Filter).
* `disparidad.py`: Funciones auxiliares para la evaluación de error vertical tras rectificar y extracción de perfiles de disparidad 2D.
* `demostracion_ssd.py`: Script visual para demostrar la búsqueda de correspondencias 1D mediante el cálculo de la Suma de Diferencias al Cuadrado (SSD).

**Calibración:**
* `calibrar.py` y `calibracion.py`: Scripts para la obtención de la matriz intrínseca K y coeficientes de distorsión utilizando un tablero de ajedrez.

## Cómo Ejecutar el Proyecto

Ejecución del pipeline completo sin ventanas GUI (Recomendado para volcar resultados en la carpeta `output/`):
```bash
python3 main.py --no-gui
```

Ejecución del pipeline mostrando la visualización 3D interactiva:
```bash
python3 main.py
```

Para realizar una nueva calibración de cámara (requiere fotos en `FotosCalibracion/`):
```bash
python3 calibrar.py
```
