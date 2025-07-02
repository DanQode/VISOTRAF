from ultralytics import YOLO
import cv2
import pandas as pd
import os
import datetime

model = YOLO("yolov8n.pt")  # Modelo ligero para detección rápida
vehicle_classes = ['car', 'bus', 'truck']

def procesar_video(video_path, output_csv=None, salto_frames=3):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    if output_csv is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"outputs/conteo_vehicular_yolo_{timestamp}.csv"

    conteo = []
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % salto_frames != 0:
            continue

        print(f"Procesando frame {frame_num}...")

        results = model(frame, verbose=False)[0]
        num_vehiculos = sum(
            1 for box in results.boxes
            if results.names[int(box.cls[0])] in vehicle_classes
        )

        conteo.append({'frame': frame_num, 'vehiculos_detectados': num_vehiculos})

    cap.release()
    df = pd.DataFrame(conteo)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Conteo guardado en: {output_csv}")
    return df
