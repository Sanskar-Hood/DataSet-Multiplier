import numpy as np
import cv2 as cv

# Try to load the images
img = cv.imread('background_output.png')  # Removed leading space
mask = cv.imread('output_mask.png', cv.IMREAD_GRAYSCALE)  # Removed leading space
if img is not None and mask is not None:
    # Proceed with inpainting if the images are loaded
    dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    # cv.imwrite('inpainted_image.png', dst)
    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
