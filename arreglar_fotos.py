import os
import cv2
import glob

def corregir_resolucion():
    carpeta = "FotosCalibracion"
    patron = os.path.join(carpeta, "*.*")
    extensiones_validas = ('.jpg', '.jpeg', '.png')
    
    print(f"Procesando fotos en {carpeta}...")

    for ruta in glob.glob(patron):
        if ruta.lower().endswith(extensiones_validas):
            img = cv2.imread(ruta)
            
            if img is not None:
                # img.shape actual será (5712, 4284, 3) -> (Alto, Ancho, Canales)
                # Al rotar 90 grados antihorario, intercambia las dimensiones
                img_horizontal = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Sobrescribe el archivo original destrozando el EXIF engañoso
                cv2.imwrite(ruta, img_horizontal)
                
                # El nuevo img_horizontal.shape será (4284, 5712, 3) -> Alto 4284, Ancho 5712
                ancho = img_horizontal.shape[1]
                alto = img_horizontal.shape[0]
                print(f"Lista: {ruta} -> Nueva resolución real: {ancho}x{alto}")

    print("Todas las fotos han sido convertidas a 5712x4284.")

if __name__ == "__main__":
    corregir_resolucion()