from ultralytics import YOLO
import cv2

# Rutas
modelo_path = "yolov8n.pt"
video_path = "data/video_interseccion.mp4"

# Cargar modelo YOLOv8
model = YOLO(modelo_path)

# Leer video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ No se pudo leer el primer frame del video.")
    exit()

# Ejecutar detección en el primer frame
results = model(frame)[0]

# Contar vehículos detectados
vehicle_classes = ['car', 'bus', 'truck']
num_vehiculos = 0

for box in results.boxes:
    clase = int(box.cls[0])
    etiqueta = model.names[clase]
    if etiqueta in vehicle_classes:
        num_vehiculos += 1

print(f"✅ Vehículos detectados: {num_vehiculos}")
