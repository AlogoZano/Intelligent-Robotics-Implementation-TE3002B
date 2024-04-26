import cv2
import numpy as np

# Función para dibujar la cuadrícula en la imagen
def draw_grid(image, rows, cols):
    height, width, _ = image.shape
    row_step = height // rows
    col_step = width // cols

    # Dibujar líneas horizontales
    for i in range(1, rows):
        cv2.line(image, (0, i * row_step), (width, i * row_step), (0, 255, 0), 1)
    
    # Dibujar líneas verticales
    for j in range(1, cols):
        cv2.line(image, (j * col_step, 0), (j * col_step, height), (0, 255, 0), 1)

    return image

# Cargar la imagen
image_path = "threshold/cuadricula2.png"  # Reemplaza "tu_imagen.jpg" con la ruta de tu imagen
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("No se pudo cargar la imagen.")
    exit()

# Dibujar la cuadrícula
rows = 49
cols = 25
image_with_grid = draw_grid(image.copy(), rows, cols)

# Mostrar la imagen original y la imagen con la cuadrícula
cv2.imshow("Original", image)
cv2.imshow("Con Cuadrícula", image_with_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
