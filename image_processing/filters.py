import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from scipy.ndimage import convolve
from scipy.signal import convolve2d

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

def add_uniform_noise(app):
    dialog = InputDialog("Uniform Noise", ["Noise Intensity:"])
    if dialog.exec_():
        intensity = float(dialog.get_input_values()[0])
        noise_filter(app.input_image, 'uniform', intensity, app)

def add_gaussian_noise(app):
    dialog = InputDialog("Gaussian Noise Parameters", ["Mean:", "Std:"])
    if dialog.exec_():
        mean, std = map(float, dialog.get_input_values())
        noise_filter(app.input_image, 'gaussian', mean, std, app)

def add_salt_and_pepper_noise(app):
    dialog = InputDialog("Salt and Pepper: ", ["Noise Ratio:"])
    if dialog.exec_():
        ratio = float(dialog.get_input_values()[0])
        noise_filter(app.input_image, 'salt_pepper', ratio, app)

def noise_filter(image, noise_type, *params, app):
    width, height = image.size
    image_gray = image.convert('L')
    noisy_image = np.array(image_gray)

    if noise_type == 'uniform':
        noise = np.random.uniform(-params[0], params[0], (height, width))
        noisy_image = np.clip(noisy_image + noise * 255, 0, 255).astype(np.uint8)
    elif noise_type == 'gaussian':
        noise = np.random.normal(params[0], params[1], (height, width))
        noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        salt_pepper = np.random.rand(height, width)
        noisy_image[salt_pepper < params[0] / 2] = 0
        noisy_image[salt_pepper > 1 - params[0] / 2] = 255

    display_processed_image(noisy_image, app.ui.filter_outputImage)

def display_processed_image(image_array, label):
    output_image = Image.fromarray(image_array)
    qt_image = QImage(output_image.tobytes(), output_image.size[0], output_image.size[1], QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qt_image)
    label.setPixmap(pixmap)

def average_filter(app):
    kernel_size = 3 if app.ui.Radio3x3Kernal.isChecked() else 5
    input_img = app.output_image if app.output_image else app.input_image
    image_gray = input_img.convert('L')
    img_array = np.array(image_gray)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    filtered_image_array = convolve2d(img_array, kernel, mode='same').astype(np.uint8)
    display_processed_image(filtered_image_array, app.ui.filter_outputImage)

def median_filter(app):
    kernel_size = 3 if app.ui.Radio3x3Kernal.isChecked() else 5
    input_img = app.output_image if app.output_image else app.input_image
    image_gray = input_img.convert('L')
    img_array = np.array(image_gray)
    height, width = img_array.shape
    filtered_image = np.zeros_like(img_array)
    pad_size = kernel_size // 2
    padded_image = np.pad(img_array, pad_size, mode='constant')
    for i in range(pad_size, height + pad_size):
        for j in range(pad_size, width + pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            filtered_image[i - pad_size, j - pad_size] = np.median(window)
    display_processed_image(filtered_image, app.ui.filter_outputImage)

def gaussian_filter(app, kernel_size=3, sigma=100):
    kernel_size = 3 if app.ui.Radio3x3Kernal.isChecked() else 5
    input_img = app.output_image if app.output_image else app.input_image
    image_gray = input_img.convert('L')
    img_array = np.array(image_gray)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    filtered_image = convolve(img_array, kernel)
    display_processed_image(filtered_image, app.ui.filter_outputImage)
