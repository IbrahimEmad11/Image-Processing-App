import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QSlider , QColorDialog, QAction, QTextEdit
from PyQt5.QtCore import QTimer,Qt, QPointF
from PyQt5.QtGui import QColor, QIcon, QCursor, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QProgressBar, QDialog, QVBoxLayout
from task1 import Ui_MainWindow
from PIL import Image , ImageFilter
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np

# pyuic5 task1.ui -o task1.py

class SignalViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  
        self.input_image = Image.new("RGB", (100, 100), color="white")

    def open_img(self):
        self.input_image = Image.open("input_image.jpg")
    
    def add_uniform_noise(self, intensity=0.1):
        width, height = self.input_image.size
        noisy_image = np.array(self.input_image)
        noise = np.random.uniform(-intensity, intensity, (height, width, 3))
        noisy_image = np.clip(noisy_image + noise * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def add_gaussian_noise(self, mean=0, std=25):
        width, height = self.input_image.size
        noisy_image = np.array(self.input_image)
        noise = np.random.normal(mean, std, (height, width, 3))
        noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def add_salt_and_pepper_noise(self, density=0.01):
        width, height = self.input_image.size
        noisy_image = np.array(self.input_image)
        salt_pepper = np.random.rand(height, width, 3)
        noisy_image[salt_pepper < density / 2] = 0
        noisy_image[salt_pepper > 1 - density / 2] = 255
        return Image.fromarray(noisy_image)
    
    def apply_average_filter(self):
        filtered_img = self.input_image.filter(ImageFilter.BLUR)
        return Image.fromarray(filtered_img)

    def apply_gaussian_filter(self, sigma=1):
        filtered_img = gaussian_filter(np.array(self.input_image), sigma=sigma)
        return Image.fromarray(filtered_img)

    def apply_median_filter(self, size=3):
        filtered_img = median_filter(np.array(self.input_image), size=size)
        return Image.fromarray(filtered_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalViewerApp()
    window.setWindowTitle("Task 1")
    # app.setWindowIcon(QIcon("img/logo.png"))
    window.resize(1250,900)
    window.show()
    sys.exit(app.exec_())