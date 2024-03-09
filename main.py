import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QSlider , QColorDialog, QAction, QTextEdit
from PyQt5.QtCore import QTimer,Qt, QPointF
from PyQt5.QtGui import QColor, QIcon, QCursor, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QProgressBar, QDialog, QVBoxLayout
from task1 import Ui_MainWindow

# pyuic5 task1.ui -o task1.py

class SignalViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalViewerApp()
    window.setWindowTitle("Task 1")
    # app.setWindowIcon(QIcon("img/logo.png"))
    window.resize(1250,900)
    window.show()
    sys.exit(app.exec_())