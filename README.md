Greetings!
This is my submission for the Internship Assignment: Computer Vision Analysis on Basketball Video

# Files in this repository

- bounding_box2.csv
  .csv file containing bounding box coordinates from the given video (recieved with training on yolov5m5.pt weights).

- mycounter.py
  python script that reads bounding box coordinates, calculates a ball's movement, checks for bounces and returns total number of bounces.

- balltracker.py
  my script to track a ball's movement accurately at all times.

- myedit.png
  a screenshot of my contribution to the official yolov5's detect.py code, used instead of entire code due to authorization problems.

- ODprocess.png
  a snap of the process of object detection with yolov5 (weights used: yolov5m6.pt)

# Approach

The objective of this assignment was to extract t meaningful insights and information from the video of a person dripping a basketball.
After throughly observing the video and understanding the instructions I understood that I had to evaluate metrics from the video such as:

- the number of times the ball bounces
- detect the switch between hands while dribbling

My initial approach was to use a pre-trained object detection model, I researched and got the best weights and congif (.yaml) files for my yolov7 model.
However due to incompatible version dependencies and methods such as "model_from_yaml" being depricated (I even did try converting to json and using model_from_json), I had to end up abandoning this approach.

For some time I did try to create my own model with my own dataset from the internet(browesd Kaggle, Huggingface and repos.) but due to a bad RAM module, my PC crashed during the training process. (not a huge problem, need to replace a RAM module)

Finally I decieded to go with yolov5's maintained library for Object Detection. I got the bounces result with my script.
A virtual enviournment had to used to install yolov5's dependencies.

# Object Detection

With this I was able to read and save the video with well identified bounding boxes.
But yolov5 would only get me so far...

This is where I had to improvise and improve the official code of yolov5.
This is the file that was modified: https://github.com/ultralytics/yolov5/blob/master/detect.py
Check 'mycode.png' for the modified code. I didn't upload this code to Github since I believe I require some permission for this from the original authors.

What the modified code actually did:

- I read the bounding box coordinates that is detected from each frame
- saved the coordinates into a list and saved it into a .csv file for the next step.

With the bbox coordinates, using some mathematics to calculate the centroid of the ball and it's movement I was able to successfully calculate the total number of bounces the ball does in the original video. I do understand that the task was to count the bounce on each step and this can be implemented successfully with some more time.

# My Constraints

- hardware issues (Bad RAM module: this played a huge roll because it lead to PC crashes on high memory consumption events such as during my model training)
- incorrect and incompatible versioning of softwares.
- limited time due to some family matters, the total time I was able to dedicate to this assignment from start (research) to implementations and documenting was just a single day (Sunday).

# Model used for Object Detection training:

- yolov5s (default)
- yolo5m6.pt

# Language and packages used:

- Python
- OpenCV
- NumPy
- csv
- argparse

# Dependencies:

- Ultralytics

- yolov5
  https://github.com/ultralytics/yolov5/

- yolov7 (interrupted)

# Commands:

For object detection run:

- python detect.py --weights X:\myobjectdetector\yolov5\yolov5m6.pt --source X:\myobjectdetector\yolov5\vid.mp4

and then for counting bounces run:

- python mycounter.py

results:
video 1/1 (1520/1520) X:\myobjectdetector\yolov5\vid.mp4: 640x384 1 person, 2 sports balls, 195.2ms
Speed: 0.1ms pre-process, 197.8ms inference, 0.6ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs\detect\exp3
(od) PS X:\myobjectdetector\yolov5> python mycounter.py
Total bounces: 88
