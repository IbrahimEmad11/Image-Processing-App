import cv2
import numpy as np

def add_high_noise(image):
    noise = np.random.normal(loc=0, scale=50, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def main():
    # Load the image
    image_path = '1.jpg'  
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Could not read the image.")
        return

    # Display original image
    cv2.imshow('Original Image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Add high noise
    noisy_image = add_high_noise(original_image)

    # Display noisy image
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
