# EuPilot CINI - YOLOv5 Man Down Tracking and Counting

This repository contains a configurable man down tracker. The detections generated by YOLOv5, one of the most popular object detection algorithm, are passed to a deepSORT algorithm that implements tracking and counting tasks.

## Code

### man_down_tracking.py 
Python script that run the man down tracking and counting algorithm.ù

### man_down
Folder containing man down script.

### deep_sort 
Folder containing the deepSORT algorithm python scripts.

### utils_ 
Folder containing useful methods for loading, storage and visualize data.

### configs
Folder containing yaml file of YOLOv5 classes and deepSORT parameters.

## Algorithm informations
Algorithm inputs:
- YOLO Model weights (such as 'yolov5s', 'yolov5l', 'yolov5x', etc..) 
- ReIdentification CNN weights (such as 'osnet_x0_25.pt', 'osnet_x0_75.pt', etc...)
- Source path (path of the video file that sould be process)

Algoritm outputs:
- Folder that contains video/videos processed

## Formats accepted
- IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
- VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

## Ultralytics GitHub Repository
https://github.com/ultralytics/yolov5