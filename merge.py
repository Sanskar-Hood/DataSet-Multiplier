import cv2

# Load two images
image1 = cv2.imread('shifted_objects_on_white_new.png')
image2 = cv2.imread('inpainted_image.png')

# Resize the images to be the same size if necessary
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Define the blending ratio (0.5 for each image gives a 50% transparency to both)
alpha = 0.32
beta = 1 - alpha

# Blend the images
blended_image = cv2.addWeighted(image1, alpha, image2, beta, 0)

# Save or display the result
cv2.imwrite('blended_image_new.png', blended_image)
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
