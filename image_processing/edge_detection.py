import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import cv2

def convolve(image, kernel):
    if image is not None:
        height, width = image.shape
        k_height, k_width = kernel.shape
        output = np.zeros_like(image)
        padded_image = np.pad(image, ((k_height // 2, k_height // 2), (k_width // 2, k_width // 2)), mode='constant')
        for i in range(height):
            for j in range(width):
                output[i, j] = np.sum(padded_image[i:i + k_height, j:j + k_width] * kernel)
        return output

def canny_edge_detection(app):
    if app.input_image is not None:
        edges = cv2.Canny(app.gray_img, 100, 200)
        display_edge_detection_result(edges, app.ui.EdgeDetection_outputImage)

def sobel_edge_detection(image):
    if image is not None:
        image = image.astype(np.float32)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        grad_x = convolve(image, kernel_x)
        grad_y = convolve(image, kernel_y)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        return gradient_magnitude.astype(np.uint8)

def perform_sobel_edge_detection(app):
    if app.gray_img is not None:
        gradient_magnitude = sobel_edge_detection(app.gray_img)
        if gradient_magnitude is not None:
            display_edge_detection_result(gradient_magnitude, app.ui.EdgeDetection_outputImage)

def roberts_edge_detection(image):
    if image is not None:
        image = image.astype(np.float32)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = convolve(image, kernel_x)
        grad_y = convolve(image, kernel_y)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        return gradient_magnitude.astype(np.uint8)

def perform_roberts_edge_detection(app):
    if app.gray_img is not None:
        gradient_magnitude = roberts_edge_detection(app.gray_img)
        if gradient_magnitude is not None:
            display_edge_detection_result(gradient_magnitude, app.ui.EdgeDetection_outputImage)

def prewitt_edge_detection(image):
    if image is not None:
        image = image.astype(np.float32)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        grad_x = convolve(image, kernel_x)
        grad_y = convolve(image, kernel_y)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        return gradient_magnitude.astype(np.uint8)

def perform_prewitt_edge_detection(app):
    if app.gray_img is not None:
        gradient_magnitude = prewitt_edge_detection(app.gray_img)
        if gradient_magnitude is not None:
            display_edge_detection_result(gradient_magnitude, app.ui.EdgeDetection_outputImage)

def display_edge_detection_result(image_array, label):
    qImg = QImage(image_array.data, image_array.shape[1], image_array.shape[0], QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qImg)
    label.setPixmap(pixmap)
