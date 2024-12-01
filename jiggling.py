import cv2
import numpy as np
import random

# Paths to the image and mask
image_path = 'football_ext.png'  # Your original image
mask_path = 'output.png'  # Your SAM-generated mask
coordinates_file = 'coordinates.txt'  # Your bounding box coordinates

# Load the image and mask
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
height, width, _ = image.shape

# Create a transparent (RGBA) background with the same size as the original image
transparent_background = np.zeros((height, width, 4), dtype=np.uint8)

# Function to randomly shift the bounding box coordinates
def random_shift_coordinates(x0, y0, x1, y1, shift_max, img_width, img_height):
    shift_x = random.randint(-shift_max, shift_max)
    shift_y = random.randint(-shift_max, shift_max)

    # Shift the coordinates, ensuring they remain within image boundaries
    new_x0 = max(0, min(int(x0 + shift_x), img_width))
    new_y0 = max(0, min(int(y0 + shift_y), img_height))
    new_x1 = max(0, min(int(x1 + shift_x), img_width))
    new_y1 = max(0, min(int(y1 + shift_y), img_height))

    return new_x0, new_y0, new_x1, new_y1

# Read the coordinates from the text file
original_coordinates = []
new_coordinates = []

with open(coordinates_file, 'r') as f:
    lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Convert the bounding box coordinates to floats, keeping metadata intact
        x0, y0, x1, y1 = map(float, parts[:4])  # First four values are the bounding box coordinates
        confidence = float(parts[4])  # Confidence score
        other_value = float(parts[5])  # Other metadata value (e.g., class label or something else)

        original_coordinates.append((x0, y0, x1, y1, confidence, other_value))

        # Randomly shift the coordinates
        shift_max = 30  # Adjust this value to control the amount of shift
        new_x0, new_y0, new_x1, new_y1 = random_shift_coordinates(x0, y0, x1, y1, shift_max, width, height)
        new_coordinates.append((new_x0, new_y0, new_x1, new_y1, confidence, other_value))

        # Extract the object from the original image using the mask and original coordinates
        object_image = cv2.bitwise_and(image[int(y0):int(y1), int(x0):int(x1)],
                                       image[int(y0):int(y1), int(x0):int(x1)],
                                       mask=mask[int(y0):int(y1), int(x0):int(x1)])

        # Convert object image to RGBA with alpha from mask
        object_image_rgba = cv2.cvtColor(object_image, cv2.COLOR_BGR2BGRA)
        object_image_rgba[:, :, 3] = mask[int(y0):int(y1), int(x0):int(x1)]

        # Paste the object into the new position on the transparent background
        transparent_background[new_y0:new_y1, new_x0:new_x1] = object_image_rgba

# Save the new image with shifted objects on a transparent background
cv2.imwrite('shifted_objects_on_transparent.png', transparent_background)

# Save the new coordinates to a text file (in the same format as the original)
with open('new_coordinates.txt', 'w') as f:
    for coord in new_coordinates:
        f.write(f"{coord[0]} {coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]}\n")

print("New image with transparent background and coordinates saved successfully.")
