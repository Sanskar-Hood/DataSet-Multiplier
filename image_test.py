from ultralytics import YOLO
model = YOLO('../../models/yolov8s.pt')

results = model('../test_files/football.png')

# Extract bounding box coordinates, confidence scores, and class IDs
bounding_boxes = results[0].boxes.xyxy  # xyxy format
confidences = results[0].boxes.conf  # confidence scores
class_ids = results[0].boxes.cls  # class IDs

# Save coordinates of person class (class ID 1) to a TXT file
with open('coordinates.txt', 'w') as file:
    for bbox, confidence, class_id in zip(bounding_boxes, confidences, class_ids):
        if class_id == 0:  # Check if the class ID is 1 (person)
            x_min, y_min, x_max, y_max = bbox.tolist()
            file.write(f"{x_min} {y_min} {x_max} {y_max} {confidence} {class_id}\n")

print("Coordinates of person class have been saved to coordinates.txt")