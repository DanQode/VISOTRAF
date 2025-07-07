from src.vision_vehicular import procesar_video
import sys
from PyQt5.QtWidgets import QApplication
from dashboard_qt import VideoDashboard  # Asegúrate de tener dashboard_qt.py en tu proyecto

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDashboard()
    window.showMaximized()  # Esto es suficiente
    sys.exit(app.exec_())

