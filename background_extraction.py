import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Initialize the SAM model
sam_checkpoint = "sam_vit_b_01ec64.pth"  # Replace with the path to your SAM checkpoint
model_type = "vit_b"  # Specify the model type (vit_h, vit_l, vit_b)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Load the image
image_path = '../../test_files/football.png'  # Path to your image
image = cv2.imread(image_path)
predictor.set_image(image)

# Load coordinates from a text file
coordinates_file = 'coordinates.txt'  # Path to your coordinates file
coordinates = []
masks_list = []

with open(coordinates_file, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # Skip header line
        parts = line.strip().split()
        input_box = np.array(parts[:4], dtype=np.float32)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks_list.append((masks[0], input_box))

# Create an empty mask for the background
background_mask = np.ones(image.shape[:2], dtype=np.uint8)

# Combine all masks to form the background mask
for mask, box in masks_list:
    background_mask[mask > 0] = 0

# Apply the background mask to the image
background_image = cv2.bitwise_and(image, image, mask=background_mask)
# Set up the figure to match the image size in pixels
h, w, _ = image.shape
fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)

# Remove any padding or axes around the image
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Display the background image without axes or borders
plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Save the figure without padding
plt.savefig('background_output.png', bbox_inches='tight', pad_inches=0)

# Show the figure
plt.show()
