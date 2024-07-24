import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap

def low_pass_filter(image, app):
    if image is None:
        return

    image_array = np.array(image)
    fft_image = np.fft.fft2(image_array)
    cutoff_freq = app.ui.horizontalSlider.value() if app.pf_hybrid_flag else app.ui.VerticalSlider.value()
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0
    fft_image_lpf = fft_image * mask
    filtered_image = np.fft.ifft2(fft_image_lpf).real.astype(np.uint8)
    app.lpf_image = Image.fromarray(filtered_image)
    if app.pf_hybrid_flag:
        display_filtered_image(app.lpf_image, app.ui.filter_outputImage)
        display_filtered_image(app.lpf_image, app.ui.pass_outputImage)
    else:
        return fft_image_lpf

def high_pass_filter(image, app):
    if image is None:
        return

    image_array = np.array(image)
    fft_image = np.fft.fft2(image_array)
    cutoff_freq = app.ui.horizontalSlider.value() if app.pf_hybrid_flag else app.ui.VerticalSlider.value()
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1
    fft_image_hpf = fft_image * mask
    filtered_image2 = np.fft.ifft2(fft_image_hpf).real.astype(np.uint8)
    app.hpf_image = Image.fromarray(filtered_image2)
    if app.pf_hybrid_flag:
        display_filtered_image(app.hpf_image, app.ui.filter_outputImage)
        display_filtered_image(app.hpf_image, app.ui.pass_outputImage)
    else:
        return fft_image_hpf

def display_filtered_image(image, label):
    qt_image = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qt_image)
    label.setPixmap(pixmap)
