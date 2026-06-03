import os
import glob
try:
    from PIL import Image
    from pillow_heif import register_heif_opener
except ImportError:
    print("Por favor, instala las librerías necesarias ejecutando:")
    print("pip install Pillow pillow-heif")
    exit(1)

# Registrar el formato HEIC para que Pillow pueda abrirlo
register_heif_opener()

def convert_heic_to_png(folder_path):
    # Buscar todos los archivos .HEIC y .heic
    search_pattern_lower = os.path.join(folder_path, "*.heic")
    search_pattern_upper = os.path.join(folder_path, "*.HEIC")
    
    heic_files = glob.glob(search_pattern_lower) + glob.glob(search_pattern_upper)
    
    if not heic_files:
        print(f"No se encontraron archivos HEIC en la carpeta '{folder_path}'.")
        return

    print(f"Se encontraron {len(heic_files)} archivos HEIC. Comenzando conversión...")

    for heic_file in heic_files:
        # Generar la ruta para el nuevo archivo .png
        base_name = os.path.splitext(heic_file)[0]
        png_file = f"{base_name}.png"
        
        try:
            # Abrir la imagen HEIC y guardarla como PNG
            image = Image.open(heic_file)
            image.save(png_file, format="PNG")
            
            # Cerrar la imagen para liberar el archivo
            image.close()
            
            # Eliminar la imagen HEIC original
            os.remove(heic_file)
            
            print(f"✔ Convertido y original eliminado: {os.path.basename(heic_file)} -> {os.path.basename(png_file)}")
        except Exception as e:
            print(f"✖ Error al procesar {os.path.basename(heic_file)}: {e}")

if __name__ == "__main__":
    carpeta = "FotosCalibracion"
    
    if not os.path.isdir(carpeta):
        print(f"Error: No se encontró el directorio '{carpeta}' en la ruta actual.")
    else:
        convert_heic_to_png(carpeta)
        print("¡Proceso finalizado!")
