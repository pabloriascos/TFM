import os
import random
import csv
import cv2
from ultralytics import YOLO
from tracker import Tracker

video_path = os.path.join('.', 'data', 'dos_1.mp4')
video_out_path = os.path.join('.', 'out.mp4')
csv_path = os.path.join('.', 'track_data.csv')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

fps = cap.get(cv2.CAP_PROP_FPS)  # Obtener los FPS del video

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame.shape[1], frame.shape[0]))

model = YOLO("D:\\TFM\\Tracker\\object-tracking-yolov8-deep-sort\\code\\best.pt")
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.7
desired_width = 800  
frame_count = 0 

# Crear el archivo CSV y escribir la cabecera
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Track_ID", "X", "Y", "Class"])
    

while ret:
    frame_data = []  # Lista para almacenar datos del frame actual
    results = model(frame)
    for result in results:
        detections = []
        classes = []  # Lista para almacenar las clases de las detecciones
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, x2, y1, y2, class_id = map(int, [x1, x2, y1, y2, class_id])
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
                classes.append(class_id)  # Añade la clase a la lista

        tracker.update(frame, detections)

        for i, track in enumerate(tracker.tracks):
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Obtiene la clase de la detección actual si está disponible
            class_name = "Unknown"
            if i < len(classes):
                class_name = "pig_lying" if classes[i] == 0 else "pig_notlying"

            frame_data.append((frame_count, track_id, center[0], center[1], class_name))

            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[track_id % len(colors)], 2)

    # Guardar datos del frame actual en el CSV
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for data in frame_data:
            writer.writerow(data)

    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    desired_height = int(desired_width / aspect_ratio)
    resized_frame = cv2.resize(frame, (desired_width, desired_height))

    cv2.imshow('frame', resized_frame)
    cap_out.write(frame)
    ret, frame = cap.read()
    frame_count += 1  # Incrementar contador de frames

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_out.release()
cv2.destroyAllWindows()
