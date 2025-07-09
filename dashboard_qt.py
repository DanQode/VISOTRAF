import sys
import cv2
import os
import shutil
import csv
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLineEdit, QGridLayout, QGroupBox, QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from src.vision_vehicular import model, vehicle_classes
from src.prediccion_AI import TrafficPredictor
from threading import Thread, Lock
import time



DIRECCIONES = ["Norte", "Sur", "Este", "Oeste"]

class VideoView(QWidget):
    def __init__(self, direccion):
        super().__init__()
        self.direccion = direccion
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.frame_num = 0

        # --- Hilos ---
        self.running = False
        self.frame_for_display = None
        self.lock = Lock()

        # Layout principal de la vista
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Widget contenedor para superponer etiqueta y video
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Etiqueta de dirección (pequeña, esquina superior izquierda)
        self.dir_label = QLabel(self.direccion[0])  # Solo la inicial
        self.dir_label.setStyleSheet(
            "background-color: #F1C40F; color: #23272A; font-weight: bold; "
            "border-radius: 6px; padding: 1px 6px; font-size: 13px;"
        )
        self.dir_label.setFixedHeight(18)
        self.dir_label.setAlignment(Qt.AlignCenter)

        # Layout horizontal para la etiqueta y espacio
        top_row = QHBoxLayout()
        top_row.setContentsMargins(4, 4, 4, 0)
        top_row.addWidget(self.dir_label, alignment=Qt.AlignLeft)
        top_row.addStretch(1)

        # Video label
        self.label = QLabel(f"{self.direccion}\n(Sin fuente)")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "background-color: #181A1B; border: 2px solid #444; border-radius: 10px; min-height: 160px;"   
        )

        container_layout.addLayout(top_row)
        container_layout.addWidget(self.label)
        container.setLayout(container_layout)
        layout.addWidget(container)

        # Layout horizontal para caja de texto y botones
        input_btn_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("URL/IP/video/imagen")
        input_btn_layout.addWidget(self.input_line)

        # Botón seleccionar archivo
        self.select_btn = QPushButton("📂")
        self.select_btn.setFixedSize(28, 28)
        self.select_btn.clicked.connect(self.select_file)
        input_btn_layout.addWidget(self.select_btn)

        # Botón iniciar
        self.start_btn = QPushButton("▶")
        self.start_btn.setFixedSize(28, 28)
        self.start_btn.clicked.connect(self.start_video)
        input_btn_layout.addWidget(self.start_btn)

        layout.addLayout(input_btn_layout)
        self.setLayout(layout)

    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Selecciona un archivo", "", "Videos/Imágenes (*.mp4 *.avi *.mov *.jpg *.png)")
        if file_path:
            self.input_line.setText(file_path)

    def start_video(self):
        source = self.input_line.text().strip()
        if not source:
            self.label.setText(f"{self.direccion}\n(Sin fuente)")
            return
        if source.lower().endswith(('.jpg', '.png')):
            self.cap = None
            img = cv2.imread(source)
            if img is not None:
                self.show_frame(img)
        else:
            self.cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
            self.running = True
            self.proc_thread = Thread(target=self.process_frames, daemon=True)
            self.proc_thread.start()
            self.timer.start(40)  # Solo para visualización

    def process_frames(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.detect_vehicles(frame)
            with self.lock:
                self.frame_for_display = processed_frame.copy()
            time.sleep(0.01)  # Pequeña pausa para no saturar CPU

    def next_frame(self):
        with self.lock:
            frame = self.frame_for_display.copy() if self.frame_for_display is not None else None
        if frame is not None:
            self.show_frame(frame)

    def stop_video(self):
        self.running = False
        if hasattr(self, 'proc_thread'):
            self.proc_thread.join(timeout=1)
        self.timer.stop()
        if self.cap:
            self.cap.release()

    def detect_vehicles(self, frame):
        results = model(frame, conf=0.4, verbose=False)[0]
        obj_id = 1
        self.count = 0  # Inicializa el conteo en cada frame
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            conf = float(box.conf[0])
            if class_name in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{obj_id}: {class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241, 196, 15), 2)
                obj_id += 1
                self.count += 1  # Suma 1 por cada vehículo detectado
        return frame

    def show_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(600, 340, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

class VideoDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISOTRAF - Dashboard de Intersección")
        self.setStyleSheet("""
            QWidget { background-color: #23272A; color: #F7F7F7; }
            QLabel#TitleLabel { font-size: 28px; font-weight: bold; color: #F1C40F; background: #23272A; padding: 16px; border-radius: 10px; }
            QGroupBox { border: 2px solid #F1C40F; border-radius: 8px; margin-top: 10px; }
            QPushButton { font-size: 14px; font-weight: bold; border-radius: 8px; padding: 4px 10px; }
        """)

        # Título y nombre de intersección
        title_layout = QHBoxLayout()
        self.title_label = QLabel("🚦 VISOTRAF 🚦")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignLeft)
        title_layout.addWidget(self.title_label)
        self.intersection_name = QLineEdit()
        self.intersection_name.setPlaceholderText("Nombre de la intersección")
        self.intersection_name.setFixedWidth(300)
        title_layout.addWidget(self.intersection_name)
        title_layout.addStretch(1)

        grid = QGridLayout()
        self.views = {}

        # 2x2 grid: Norte (0,0), Este (0,1), Oeste (1,0), Sur (1,1)
        self.views["Norte"] = VideoView("Norte")
        grid.addWidget(self.views["Norte"], 0, 0)
        self.views["Este"] = VideoView("Este")
        grid.addWidget(self.views["Este"], 0, 1)
        self.views["Oeste"] = VideoView("Oeste")
        grid.addWidget(self.views["Oeste"], 1, 0)
        self.views["Sur"] = VideoView("Sur")
        grid.addWidget(self.views["Sur"], 1, 1)

        # Panel izquierdo: título y grid
        main_layout = QVBoxLayout()
        main_layout.addLayout(title_layout)
        main_layout.addLayout(grid)

        # Panel derecho: botones y dos frames informativos alargados y delgados
        action_layout = QVBoxLayout()
        self.global_start_btn = QPushButton("Iniciar todos")
        self.global_start_btn.setFixedSize(110, 32)
        self.global_start_btn.clicked.connect(self.iniciar_todos)
        action_layout.addWidget(self.global_start_btn)
        self.global_stop_btn = QPushButton("Detener todos")
        self.global_stop_btn.setFixedSize(110, 32)
        self.global_stop_btn.clicked.connect(self.detener_todos)
        action_layout.addWidget(self.global_stop_btn)
        self.export_btn = QPushButton("Exportar histórico")
        self.export_btn.setFixedSize(140, 32)
        self.export_btn.clicked.connect(self.exportar_historico)
        action_layout.addWidget(self.export_btn)

        # Frame: Conteo de vehículos (delgado y alargado)
        self.count_box = QGroupBox("Conteo de vehículos")
        self.count_box.setMinimumWidth(180)
        self.count_box.setMaximumWidth(200)
        self.count_box.setMinimumHeight(300)
        self.count_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        count_layout = QVBoxLayout()
        self.count_label = QLabel("Aquí irá el conteo de vehículos por dirección.")
        self.count_label.setWordWrap(True)
        count_layout.addWidget(self.count_label)
        self.count_box.setLayout(count_layout)
        action_layout.addWidget(self.count_box)

        # Frame: Resultado VISOTRAF (delgado y alargado)
        self.result_box = QGroupBox("Resultado VISOTRAF")
        self.result_box.setMinimumWidth(180)
        self.result_box.setMaximumWidth(200)
        self.result_box.setMinimumHeight(300)
        self.result_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        result_layout = QVBoxLayout()
        self.result_label = QLabel("Aquí irá el resultado del análisis VISOTRAF.")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        self.result_box.setLayout(result_layout)
        action_layout.addWidget(self.result_box)

        action_layout.addStretch(1)

        # Layout horizontal principal
        h_layout = QHBoxLayout()
        h_layout.addLayout(main_layout, 4)
        h_layout.addLayout(action_layout, 1)
        self.setLayout(h_layout)

        self.predictor = TrafficPredictor()
        # Timer para actualización de conteo
        self.tiempo_restante = 10
        self.timer_conteo = QTimer()
        self.timer_conteo.timeout.connect(self.actualizar_conteo_vehiculos)
        self.timer_conteo.start(1000)
        self.actualizar_conteo_vehiculos(inicial=True)

        # Temporizador dedicado para guardar conteos cada 10 segundos
        self.timer_guardado = QTimer()
        self.timer_guardado.timeout.connect(self.guardar_conteo_periodico)
        self.timer_guardado.start(10000)  # 10,000 ms = 10 segundos

        
    def iniciar_todos(self):
        for view in self.views.values():
            view.start_video()

    def detener_todos(self):
        for view in self.views.values():
            view.stop_video()
            view.label.setText(f"{view.direccion}\n(Detenido)")

    def guardar_conteo_periodico(self):
        conteos = {
            "Norte": getattr(self.views["Norte"], "count", 0),
            "Sur": getattr(self.views["Sur"], "count", 0),
            "Este": getattr(self.views["Este"], "count", 0),
            "Oeste": getattr(self.views["Oeste"], "count", 0),
        }
        self.guardar_conteo_csv(conteos)

    def guardar_conteo_csv(self, conteos):
        archivo = "conteo_vehiculos.csv"
        existe = os.path.isfile(archivo)
        with open(archivo, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not existe:
                writer.writerow(["timestamp", "Norte", "Sur", "Este", "Oeste"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                conteos["Norte"],
                conteos["Sur"],
                conteos["Este"],
                conteos["Oeste"]
            ])

    def actualizar_conteo_vehiculos(self, inicial=False):
        if not inicial:
            self.tiempo_restante -= 1
        if self.tiempo_restante <= 0 or inicial:
            conteos = {
                "Norte": getattr(self.views["Norte"], "count", 0),
                "Sur": getattr(self.views["Sur"], "count", 0),
                "Este": getattr(self.views["Este"], "count", 0),
                "Oeste": getattr(self.views["Oeste"], "count", 0),
            }
            self.guardar_conteo_csv(conteos)
            texto = (
                f"Conteo de vehículos (actualizado):\n"
                f"Norte: {conteos['Norte']}\n"
                f"Sur: {conteos['Sur']}\n"
                f"Este: {conteos['Este']}\n"
                f"Oeste: {conteos['Oeste']}\n"
            )
            self.count_label.setText(texto)
            # --- Predicción de tiempos de semáforo ---
            predictions, cycle_sequence = self.predictor.predict_green_times(conteos)
            total_cycle_time = sum(phase['duration'] for phase in cycle_sequence)
            texto_prediccion = (
                f"Predicción de tiempos:\n\n"
                f"Norte-Sur:\n"
                f"  Principal: {predictions['main']['ns']:.1f}s\n"
                f"  Giro: {predictions['turn']['ns']:.1f}s\n\n"
                f"Este-Oeste:\n"
                f"  Principal: {predictions['main']['eo']:.1f}s\n"
                f"  Giro: {predictions['turn']['eo']:.1f}s\n\n"
                f"Tiempo total ciclo: {total_cycle_time:.1f}s"
            )
            self.result_label.setText(texto_prediccion)
            # -----------------------------------------
            self.tiempo_restante = 10
        self.count_box.setTitle(f"Conteo de vehículos - {self.tiempo_restante}s")
        

    def exportar_historico(self):
        archivo_origen = "conteo_vehiculos.csv"
        if not os.path.exists(archivo_origen):
            QMessageBox.warning(self, "Exportar histórico", "No existe archivo de datos históricos.")
            return
        # Genera nombre con fecha y hora
        fecha_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"conteo_vehiculos_export_{fecha_hora}.csv"
        destino, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar histórico como...",
            nombre_archivo,
            "CSV Files (*.csv)"
        )
        if destino:
            shutil.copyfile(archivo_origen, destino)
            QMessageBox.information(self, "Exportar histórico", f"Archivo exportado correctamente como:\n{os.path.basename(destino)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDashboard()
    window.showMaximized()
    sys.exit(app.exec_())