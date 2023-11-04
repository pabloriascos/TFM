import json
from PIL import Image
import os

# Clases
classes = {
    "pigLying": 0,
    "pigNotLying": 1
}

# Leer el archivo JSON
input_file = r'D:\TFM\COMPAG_105391_Riekert_etal_2020\train\annotations_mr_2018-06-22_n-lying.json'

print("Abriendo el archivo JSON...")  # Mensaje de diagnóstico
with open(input_file, 'r') as file:
    data = json.load(file)
print("Archivo JSON cargado.")  # Mensaje de diagnóstico

# Ruta a las imágenes
images_path = r'D:\TFM\COMPAG_105391_Riekert_etal_2020\train'

# Procesar cada imagen en el JSON
for image in data:
    print(f"Procesando imagen: {image['filename']}...")  # Mensaje de diagnóstico
    
    # Obtener el nombre del archivo de la imagen y crear un nombre para el archivo de salida
    image_filename = image['filename']
    output_filename = image_filename.split('.')[0] + '.txt'  # Cambia la extensión a .txt
    
    # Obtener las dimensiones de la imagen
    image_path = os.path.join(images_path, image_filename)
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"No se pudo abrir la imagen {image_path}. Error: {e}")
        continue  # Saltar al siguiente ciclo si no se pudo abrir la imagen
    
    # Procesar las anotaciones de la imagen
    annotations = []
    for annotation in image['annotations']:
        # Obtener la clase y las coordenadas del bounding box
        class_id = classes[annotation['class']]
        x_center = (annotation['x'] + annotation['width'] / 2) / width
        y_center = (annotation['y'] + annotation['height'] / 2) / height
        norm_width = annotation['width'] / width
        norm_height = annotation['height'] / height
        
        # Formatear la anotación y añadirla a la lista
        annotations.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
    
    # Escribir las anotaciones en el archivo de salida
    with open(output_filename, 'w') as file:
        file.write('\n'.join(annotations))

print("Proceso completado.")
