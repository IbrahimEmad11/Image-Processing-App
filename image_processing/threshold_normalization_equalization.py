import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton

class InputDialog(QDialog):
    def __init__(self, title, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        self.inputs = []
        for label_text in labels:
            label = QLabel(label_text)
            line_edit = QLineEdit()
            layout.addWidget(label)
            layout.addWidget(line_edit)
            self.inputs.append(line_edit)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        layout.addWidget(self.apply_button)
        self.setLayout(layout)
    
    def get_input_values(self):
        return [input_field.text() for input_field in self.inputs]

    def apply_action(self):
        self.accept()

def image_equalization(app):
    histogram, bins = np.histogram(app.gray_img.flatten(), 256, [0, 256])
    cdf1 = histogram.cumsum()
    cdf_m = np.ma.masked_equal(cdf1, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    eq_img = cdf[app.gray_img]
    equalized_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)
    display_processed_image(equalized_img, app.ui.Threshold_outputImage)

def image_normalization(app):
    dialog = InputDialog("Global Threshold Parameter (0-255): ", ["Value:"])
    if dialog.exec_():
        threshold = float(dialog.get_input_values()[0])
    lmin = float(app.gray_img.min())
    lmax = float(app.gray_img.max())
    normalized_img = ((app.gray_img - lmin) / (lmax - lmin)) * threshold
    display_processed_image(normalized_img, app.ui.Threshold_outputImage)

def global_threshold(app):
    dialog = InputDialog("Global Threshold Parameter (0-255): ", ["Value:"])
    if dialog.exec_():
        threshold = float(dialog.get_input_values()[0])
    thresholded_image = np.zeros_like(app.gray_img)
    thresholded_image[app.gray_img > threshold] = 220
    display_processed_image(thresholded_image, app.ui.Threshold_outputImage)

def local_threshold(app):
    block_size = 3 if app.ui.Radio3x3Kernal_2.isChecked() else 5
    dialog = InputDialog("Global Threshold Parameter (0-255): ", ["Value:"])
    if dialog.exec_():
        threshold = float(dialog.get_input_values()[0])
    thresholded_image = np.zeros_like(app.gray_img)
    padded_image = np.pad(app.gray_img, block_size // 2, mode='constant')
    height, width = app.gray_img.shape
    for i in range(app.gray_img.shape[0]):
        for j in range(app.gray_img.shape[1]):
            neighborhood = padded_image[i:i + block_size, j:j + block_size]
            mean_value = np.mean(neighborhood)
            thresholded_image[i, j] = 255 if (app.gray_img[i, j] - mean_value) > threshold else 0
    display_processed_image(thresholded_image, app.ui.Threshold_outputImage)

def display_processed_image(image_array, label):
    qImg = QImage(image_array.data, image_array.shape[1], image_array.shape[0], QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qImg)
    label.setPixmap(pixmap)
