import numpy as np
from PIL import Image

def global_threshold(image, threshold, label):
    thresholded_image = np.zeros_like(image)
    thresholded_image[image > threshold] = 255
    
    thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB)
    height, width, channel = thresholded_image.shape
    bytesPerLine = 3 * width
    qImg = QImage(thresholded_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qImg)
    label.setPixmap(pixmap)

def local_threshold(image, block_size, constant, label):
    thresholded_image = np.zeros_like(image)
    padded_image = np.pad(image, block_size//2, mode='constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+block_size, j:j+block_size]
            mean_value = np.mean(neighborhood)
            thresholded_image[i, j] = 255 if (image[i, j] - mean_value) > constant else 0
    
    thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB)
    height, width, channel = thresholded_image.shape
    bytesPerLine = 3 * width
    qImg = QImage(thresholded_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qImg)
    label.setPixmap(pixmap)