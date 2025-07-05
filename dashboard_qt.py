import sys
import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QRadioButton, QButtonGroup, QTextEdit, QLineEdit
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

from src.vision_vehicular import procesar_video

class VideoDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISOTRAF - Dashboard Vehicular Interactivo")
        self.setStyleSheet("""
            QWidget {
                background-color: #23272A;
                color: #F7F7F7;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #F1C40F;
                background: #23272A;
                padding: 16px;
                border-radius: 10px;
            }
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton#Start {
                background-color: #27AE60;
                color: white;
            }
            QPushButton#Select {
                background-color: #2980B9;
                color: white;
            }
            QPushButton#Back {
                background-color: #E74C3C;
                color: white;
            }
            QPushButton#Update {
                background-color: #F1C40F;
                color: #23272A;
            }
            QTextEdit, QLineEdit {
                background-color: #2C2F33;
                color: #F7F7F7;
                border: 2px solid #444;
                border-radius: 8px;
                font-size: 15px;
            }
            QRadioButton {
                font-size: 15px;
            }
        """)

        self.video_path = None
        self.df = None
        self.cap = None
        self.fps = 30
        self.frame_num = 0
        self.analizando = False

        # Layouts
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Título principal VISOTRAF
        self.title_label = QLabel("🚦 VISOTRAF 🚦")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #F1C40F; background: #23272A; padding: 16px; border-radius: 10px;"
        )
        main_layout.insertWidget(0, self.title_label, 1)

        # Video label
        self.video_label = QLabel("Selecciona una opción para comenzar")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #181A1B; border: 2px solid #444; border-radius: 10px; min-height: 320px;")
        left_layout.addWidget(self.video_label)

        # Radio buttons para elegir fuente
        self.radio_group = QButtonGroup(self)
        self.radio_video = QRadioButton("🎞️ Video pregrabado")
        self.radio_live = QRadioButton("📷 Video en vivo (cámara)")
        self.radio_group.addButton(self.radio_video)
        self.radio_group.addButton(self.radio_live)
        left_layout.addWidget(self.radio_video)
        left_layout.addWidget(self.radio_live)

        # Botón de selección de video
        self.select_btn = QPushButton("Seleccionar archivo de video")
        self.select_btn.setObjectName("Select")
        self.select_btn.clicked.connect(self.select_video)
        left_layout.addWidget(self.select_btn)

        # Botón de iniciar análisis
        self.start_btn = QPushButton("Iniciar análisis")
        self.start_btn.setObjectName("Start")
        self.start_btn.clicked.connect(self.start_analysis)
        left_layout.addWidget(self.start_btn)

        # Botón de retroceder
        self.back_btn = QPushButton("Retroceder")
        self.back_btn.setObjectName("Back")
        self.back_btn.clicked.connect(self.retroceder)
        left_layout.addWidget(self.back_btn)

        # Botón de actualizar
        self.update_btn = QPushButton("Actualizar")
        self.update_btn.setObjectName("Update")
        self.update_btn.clicked.connect(self.actualizar)
        left_layout.addWidget(self.update_btn)

        # Entrada para URL de cámara IP
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("URL de cámara IP (opcional)")
        left_layout.addWidget(self.url_input)

        left_layout.addStretch(1)

        # Conteo de autos
        self.count_label = QLabel("🚗 Conteo de autos por frame:")
        self.count_label.setStyleSheet("font-size: 18px; color: #F1C40F; font-weight: bold;")
        right_layout.addWidget(self.count_label)
        self.count_text = QTextEdit()
        self.count_text.setReadOnly(True)
        right_layout.addWidget(self.count_text)

        # Espacio para recomendaciones (puedes personalizarlo)
        self.recommend_label = QLabel("💡 Recomendación de semáforo:")
        self.recommend_label.setStyleSheet("font-size: 18px; color: #58D68D; font-weight: bold;")
        right_layout.addWidget(self.recommend_label)
        self.recommend_text = QTextEdit("Aquí aparecerán recomendaciones según el flujo vehicular.")
        self.recommend_text.setReadOnly(True)
        right_layout.addWidget(self.recommend_text)

        right_layout.addStretch(1)

        top_layout.addLayout(left_layout, 2)
        top_layout.addLayout(right_layout, 1)
        main_layout.addLayout(top_layout)
        self.setLayout(main_layout)

        # Timer para actualizar frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Estado de selección
        self.radio_video.setChecked(True)
        self.select_btn.setEnabled(True)
        self.radio_video.toggled.connect(self.toggle_video_selection)
        self.radio_live.toggled.connect(self.toggle_video_selection)

    def toggle_video_selection(self):
        if self.radio_video.isChecked():
            self.select_btn.setEnabled(True)
        else:
            self.select_btn.setEnabled(False)
            self.video_path = 0  # Cámara en vivo

    def select_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Selecciona un video", "", "Videos (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_path = video_path
            self.video_label.setText(f"Video seleccionado:\n{video_path}")

    def start_analysis(self):
        if self.analizando:
            return
        self.analizando = True
        self.frame_num = 0
        self.count_text.clear()
        self.recommend_text.clear()
        if self.radio_live.isChecked():
            url = self.url_input.text().strip()
            if url:
                self.cap = cv2.VideoCapture(url)
            else:
                self.cap = cv2.VideoCapture(0)
            self.fps = 30
            self.df = pd.DataFrame(columns=['frame', 'vehiculos_detectados'])
        elif self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.df = procesar_video(self.video_path, visualizar=False)
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.video_label.setText("Selecciona un video primero.")
            self.analizando = False
            return
        self.timer.start(int(1000 / self.fps))

    def retroceder(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.analizando = False
        self.video_label.setText("Selecciona una opción para comenzar")
        self.count_text.clear()
        self.recommend_text.clear()
        self.radio_video.setChecked(True)
        self.video_path = None
        self.df = None
        self.frame_num = 0

    def actualizar(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.analizando = False
        self.start_analysis()

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.analizando = False
            self.video_label.setText("Fin del video o cámara desconectada.")
            return
        self.frame_num += 1

        from src.vision_vehicular import model, vehicle_classes

        if self.radio_live.isChecked():
            results = model(frame, conf=0.4, verbose=False)[0]
            num_vehiculos = 0
            obj_id = 1
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                conf = float(box.conf[0])
                if class_name in vehicle_classes:
                    num_vehiculos += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{obj_id}: {class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (241, 196, 15), 2)
                    obj_id += 1
            self.df.loc[len(self.df)] = [self.frame_num, num_vehiculos]
        else:
            results = model(frame, conf=0.4, verbose=False)[0]
            num_vehiculos = 0
            obj_id = 1
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                conf = float(box.conf[0])
                if class_name in vehicle_classes:
                    num_vehiculos += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{obj_id}: {class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (241, 196, 15), 2)
                    obj_id += 1
            cv2.putText(
                frame,
                f"Autos detectados: {num_vehiculos}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (46, 204, 113),
                3
            )

        # Mostrar frame
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # Actualizar conteo en panel derecho
        self.count_text.append(f"Frame {self.frame_num}: {num_vehiculos} autos")

        # Recomendación simple de semáforo
        if self.frame_num % self.fps == 0:
            promedio = self.df['vehiculos_detectados'].tail(self.fps).mean() if not self.df.empty else 0
            if promedio < 3:
                recomendacion = "🟢 Tiempo verde sugerido: 10 segundos"
            elif promedio < 7:
                recomendacion = "🟡 Tiempo verde sugerido: 20 segundos"
            else:
                recomendacion = "🔴 Tiempo verde sugerido: 30 segundos"
            self.recommend_text.setText(
                f"Promedio de autos recientes: {promedio:.2f}\n{recomendacion}"
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDashboard()
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec_())