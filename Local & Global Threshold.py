import numpy as np
from PIL import Image

def global_threshold(image, threshold):
    thresholded_image = np.zeros_like(image)
    thresholded_image[image > threshold] = 255
    return thresholded_image

def local_threshold(image, block_size, constant):
    thresholded_image = np.zeros_like(image)
    padded_image = np.pad(image, block_size//2, mode='constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+block_size, j:j+block_size]
            mean_value = np.mean(neighborhood)
            thresholded_image[i, j] = 255 if (image[i, j] - mean_value) > constant else 0
    
    return thresholded_image

# Load the image
image = Image.open('image.jpg').convert('L')  # Convert to grayscale
image = np.array(image)

# Apply global thresholding
global_threshold_value = 128  # Adjust this threshold value as needed
global_threshold_image = global_threshold(image, global_threshold_value)

# Apply local thresholding
local_block_size = 11
local_constant = 5
local_threshold_image = local_threshold(image, local_block_size, local_constant)

# Display the original and thresholded images
Image.fromarray(image).show()
Image.fromarray(global_threshold_image).show()
Image.fromarray(local_threshold_image).show()
