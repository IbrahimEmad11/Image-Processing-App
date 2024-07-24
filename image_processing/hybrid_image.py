import numpy as np
import cv2
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
import image_processing.hp_lp_filters as hp_lp_filters
from PyQt5.QtWidgets import QMessageBox

def generate_hybrid_image(input1, input2, app):
    if input1 is None or input2 is None:
        QMessageBox.warning(app, "Warning", "Please load two images first.")
        return

    app.pf_hybrid_flag = False
    lowpass_image1 = hp_lp_filters.low_pass_filter(input1, app)
    lowpass_image2 = hp_lp_filters.low_pass_filter(input2, app)
    highpass_image1 = hp_lp_filters.high_pass_filter(input1, app)
    highpass_image2 = hp_lp_filters.high_pass_filter(input2, app)
    highpass_image2_resized = resize_fft(highpass_image2, lowpass_image1.shape)
    highpass_image1_resized = resize_fft(highpass_image1, lowpass_image2.shape)
    hybrid_image1_fft_array = lowpass_image1 + highpass_image2_resized
    hybrid_image2_fft_array = lowpass_image2 + highpass_image1_resized
    inverse_hybrid1 = np.fft.ifft2(hybrid_image1_fft_array).real.astype(np.uint8)
    inverse_hybrid2 = np.fft.ifft2(hybrid_image2_fft_array).real.astype(np.uint8)
    app.hybrid_image1 = Image.fromarray(inverse_hybrid1)
    app.hybrid_image2 = Image.fromarray(inverse_hybrid2)
    display_hybrid_image(app)

def resize_fft(image_fft, target_shape):
    real_part = np.real(image_fft)
    imag_part = np.imag(image_fft)
    real_resized = cv2.resize(real_part, (target_shape[1], target_shape[0]))
    imag_resized = cv2.resize(imag_part, (target_shape[1], target_shape[0]))
    return real_resized + 1j * imag_resized

def display_hybrid_image(app):
    if app.ui.radioButton_2.isChecked():
        display_processed_image(app.hybrid_image1, app.ui.finalHybridImage)
    else:
        display_processed_image(app.hybrid_image2, app.ui.finalHybridImage)
    app.pf_hybrid_flag = True

def display_processed_image(image, label):
    qt_image = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qt_image)
    label.setPixmap(pixmap)
