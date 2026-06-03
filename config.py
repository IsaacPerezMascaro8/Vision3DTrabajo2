"""
Pequeño módulo de configuración para controlar la salida por consola
y el comportamiento GUI (mostrar ventanas). Usado por los demás módulos
para silenciar mensajes fácilmente con --quiet o deshabilitar ventanas
con --no-gui.
"""
import os

# Flags por defecto
VERBOSE = True
SHOW_GUI = True


def set_verbose(v: bool):
    global VERBOSE
    VERBOSE = bool(v)


def set_show_gui(v: bool):
    global SHOW_GUI
    SHOW_GUI = bool(v)


def info(msg: str = ""):
    """Imprime un mensaje informativo cuando VERBOSE == True."""
    if VERBOSE:
        print(msg)


def separator(titulo: str):
    """Imprime un separador bonito cuando VERBOSE == True."""
    if not VERBOSE:
        return
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def ensure_output_dir(path: str = "output"):
    """Crea el directorio de salida si no existe y devuelve la ruta."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    """Guarda un objeto serializable como JSON en `path`."""
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
