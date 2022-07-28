# EuPilot CINI - YOLO Man Down Detection

## Scripts:
1) man_down.py --> Python script that implement the man down algorithm
2) YOLOv5_PyTorch.py --> Python script that implement the detection layer using YOLOv5 pretrained on COCO dataset
3) man_down_test.py --> Python testing script 

## Algorithm informations
Algorithm inputs:
1) Model weights (such as 'yolov5s', 'yolov5l', 'yolov5x' ecc..) 
2) Classes dictionary path
3) Source path (path of the image or video file that sould be process)
4) Output path 
5) Name of the directory where the output files is saved

Algoritm outputs:
1) Folder that contains image/images or video/videos processed

## Formats accepted
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

## Link Ultralytics Github Repository
https://github.com/ultralytics/yolov5