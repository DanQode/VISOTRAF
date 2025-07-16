# VISOTRAF

VISOTRAF es un sistema inteligente para la gestión y monitoreo de intersecciones viales mediante visión computacional y predicción automática de tiempos semafóricos.

---

## 🚦 **Nuevas Funcionalidades**

### 1. **Predicción automática de tiempos de semáforo con LSTM**
- Integración de un modelo LSTM entrenado para sugerir tiempos óptimos de luz verde para cada fase del semáforo.
- El sistema toma en cuenta el conteo de vehículos por dirección y variables temporales (hora y día).
- Los tiempos sugeridos se actualizan dinámicamente y se muestran en la interfaz.

### 2. **Conteo de vehículos en tiempo real**
- Detección y conteo automático de vehículos por dirección (Norte, Sur, Este, Oeste) usando modelos de visión (YOLOv8).
- Visualización en tiempo real del conteo en el dashboard.
- Almacenamiento histórico de los conteos en un archivo CSV.

### 3. **Exportación de histórico**
- Opción para exportar el histórico de conteo de vehículos a un archivo CSV desde la interfaz gráfica.
- Al exportar, se imprime en consola el contenido del histórico.

### 4. **Interfaz gráfica mejorada**
- Visualización clara de los conteos por dirección y los tiempos sugeridos por el modelo.
- Botones para iniciar/detener todos los videos y exportar el histórico.
- Paneles informativos y visualización de secuencias de fases semafóricas.

### 5. **Algoritmo de fallback**
- Si el modelo LSTM no está disponible, el sistema utiliza un algoritmo basado en reglas para sugerir tiempos de semáforo.

---

## 🛠️ **Requisitos**

- Python 3.11.9 #La versión depende si tensorflow tiene soporte
- TensorFlow 2.6
- PyQt5
- OpenCV
- numpy, pandas, scikit-learn

---

## 🚀 **Ejecución**

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el dashboard:
   ```bash
   python main.py
   ```

---

## 📁 **Estructura del Proyecto**

- `main.py` — Lanza la interfaz principal.
- `dashboard_qt.py` — Lógica y GUI del dashboard.
- `src/vision_vehicular.py` — Detección y conteo de vehículos.
- `src/prediccion_LSTM.py` — Predicción de tiempos con LSTM.
- `models/lstm_traffic_light_model*.h5` — Modelos LSTM entrenados.
- `conteo_vehiculos.csv` — Histórico de conteos.

---

## ✨ **Notas**

- El archivo `test_yolo_frame.py` es solo para probar que yolo8n.pt (un modelo menos pesado que yolo8m.pt) funciona correctamente sobre videos y no interviene en el flujo principal. 
- El sistema puede funcionar sin el modelo LSTM, usando reglas simples para la predicción.

---

**Desarrollado para la gestión inteligente de tráfico urbano.**
