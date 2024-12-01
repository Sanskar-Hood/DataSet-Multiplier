import cv2
import numpy as np

# Load the image (replace with your actual file path)
image_path = 'background_output.png'
image = cv2.imread(image_path)

# Convert to grayscale, so we can easily manipulate the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to create a mask where black parts are white (255) and others are black (0)
_, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)

height, width = image.shape[:2]
mask = cv2.resize(mask, (width, height))
# Save the result
cv2.imwrite('inverted_mask.png', mask)

# Display the mask (optional)
# cv2.imshow("Inverted Mask", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
