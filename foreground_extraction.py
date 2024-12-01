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


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


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
image_path = 'shifted_objects_on_white.png'  # Path to your image
image = cv2.imread(image_path)
predictor.set_image(image)

# Load coordinates from a text file
coordinates_file = 'new_coordinates.txt'  # Path to your coordinates file
coordinates = []
masks_list =[]
with open(coordinates_file, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # Skip header line
        parts = line.strip().split()
        input_box= np.array(parts[:4])

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks_list.append((masks[0], input_box))
# Set up the figure to match the image size in pixels
h, w, _ = image.shape
fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)

# Remove any padding or axes around the image
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax = plt.gca()

# Display all masks and boxes on the image
for mask, box in masks_list:
    show_mask(mask, ax)
    # show_box(box, ax)  # Uncomment if you want to show boxes

plt.axis('off')

# Save the image without extra padding
plt.savefig('output2.png', bbox_inches='tight', pad_inches=0)

# Show the figure
plt.show()