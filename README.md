# EuPilot CINI - YOLO Man Down Tracking and Counting

## Scripts
1) man_down.py --> Python script that implement the man down algorithm
2) YOLOv5_PyTorch.py --> Python script that implement the detection layer using YOLOv5 pretrained on COCO dataset
3) utilities.py --> Python script that contains useful methods such as NMS, IoU, ecc... 
4) man_down_test.py --> Python testing script
5) classes.json --> JSON file which includes YOLO classes

## Algorithm informations
Algorithm inputs:
- Model weights (such as 'yolov5s', 'yolov5l', 'yolov5x' ecc..) 
- Classes dictionary path
- Source path (path of the image or video file that sould be process)
- Output path 
- Name of the directory where the output files is saved

Algoritm outputs:
- Folder that contains image/images or video/videos processed

## Formats accepted
- IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
- VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

## Ultralytics GitHub Repository
https://github.com/ultralytics/yolov5