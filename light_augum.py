import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('shifted_objects_on_white.png')

# Define a function to apply light augmentations
def apply_light_augmentation(image):
    rows, cols, _ = image.shape

    # 1. Light Rotation
    rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), np.random.uniform(-5, 5), 1)  # Rotate by max ±5 degrees
    rotated_image = cv.warpAffine(image, rotation_matrix, (cols, rows))

    # 2. Small Translation (shifting)
    translation_matrix = np.float32([[1, 0, np.random.uniform(-10, 10)], [0, 1, np.random.uniform(-10, 10)]])  # Shift ±10 pixels
    translated_image = cv.warpAffine(rotated_image, translation_matrix, (cols, rows))

    # 3. Brightness/Contrast Adjustment
    brightness = np.random.uniform(0.9, 1.1)  # Brightness change between 90% and 110%
    contrast = np.random.uniform(0.9, 1.1)    # Contrast change between 90% and 110%
    adjusted_image = cv.convertScaleAbs(translated_image, alpha=contrast, beta=brightness)

    return adjusted_image

# Apply the augmentation
augmented_image = apply_light_augmentation(image)

# Save and display the augmented image
cv.imwrite('/mnt/data/augmented_image.png', augmented_image)
cv.imshow('Augmented Image', augmented_image)
cv.waitKey(0)
cv.destroyAllWindows()
