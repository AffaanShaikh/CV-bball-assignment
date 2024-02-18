import cv2
import numpy as np
import csv

# Load bounding box coordinates from the CSV file
bounding_boxes_list = []
with open('bounding_boxes.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        bounding_boxes_list.append(tuple(map(int, row)))

bounce_counter = 0
previous_height = None

# Loop through each frame
for bbox in bounding_boxes_list:
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    
    # Calculate centroid of a bounding box
    centroid_x = (bbox_x1 + bbox_x2) / 2
    centroid_y = (bbox_y1 + bbox_y2) / 2
    
    if bbox_x2 - bbox_x1 > 50 and bbox_y2 - bbox_y1 > 50:
        continue
    
    # a bounce event check
    if previous_height is not None:
        if centroid_y > previous_height:  # Ball is moving downwards
            bounce_counter += 1
    previous_height = centroid_y
    
    # display frames with bounding boxes
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)  
    cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)  
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

# Print total number of bounces
print("Total bounces:", bounce_counter)

# Release video capture and close windows
cv2.destroyAllWindows()
