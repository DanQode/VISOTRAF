from ultralytics import YOLO
import cv2
import pandas as pd
import os
import datetime

model = YOLO("yolov8m.pt")  # Modelo más preciso
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']  # Más clases

def procesar_video(video_path, output_csv=None, salto_frames=3, visualizar=False):
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

        results = model(frame, conf=0.4, verbose=False)[0]  # Ajuste de umbral
        num_vehiculos = 0

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            conf = float(box.conf[0])
            if class_name in vehicle_classes:
                num_vehiculos += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        conteo.append({'frame': frame_num, 'vehiculos_detectados': num_vehiculos})

        if visualizar:
            cv2.imshow('Detección de Vehículos', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(conteo)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Conteo guardado en: {output_csv}")
    return df
