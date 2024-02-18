import cv2
import tensorflow as tf
from tensorflow.python.keras.models import model_from_yaml
import numpy as np

# Load model configuration from YAML file
with open(r'X:\bballanalysis\yolov7.yaml', 'r') as f:
    model_config = f.read()

model = model_from_yaml(model_config) # DEPRICATED

checkpoint_dir = 'X:\bballanalysis\yolov7' 
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

latest_checkpoint = checkpoint_manager.latest_checkpoint
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Model restored from checkpoint:", latest_checkpoint)
else:
    raise FileNotFoundError("Checkpoint not found. Please check the directory path.")

cap = cv2.VideoCapture('X:\bballanalysis\WHATSAAP ASSIGNMENT.mp4')

dribble_count = 0
prev_ball_position = None
hand_switch_count = 0
prev_hand_position = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection inference
    input_image = cv2.resize(frame, (model.input_shape[1], model.input_shape[0]))
    input_image = input_image / 255.0  # Normalize pixel values to [0, 1]
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    predictions = model.predict(input_image)

    # Analyze predictions for dribble count and hand switch detection
    for prediction in predictions:
        for obj in prediction:
            class_id = int(obj[5])
            if class_id == 0:  # Assuming class 0 corresponds to basketball
                ball_position = obj[:4]
                if prev_ball_position is not None:
                    # Check for dribble
                    if ball_position[3] - prev_ball_position[3] > 0.5:  # Adjust threshold as needed
                        dribble_count += 1
                prev_ball_position = ball_position
            elif class_id == 1:  # Assuming class 1 corresponds to player's hand
                hand_position = obj[:4]
                if prev_hand_position is not None:
                    # Check for hand switch
                    if hand_position[2] < prev_hand_position[2]:  # Assuming left-right movement for hand switch
                        hand_switch_count += 1
                prev_hand_position = hand_position

    # Display dribble count and hand switch count on frame
    cv2.putText(frame, f'Dribbles: {dribble_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Hand Switches: {hand_switch_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
