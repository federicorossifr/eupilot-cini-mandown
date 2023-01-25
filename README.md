# EuPilot CINI - YOLOv5 Man Down Tracking

This repository contains a configurable man down tracker. The detections generated by YOLOv5, one of the most popular object detection algorithm, are passed to the DeepSORT algorithm that implements tracking and counting tasks.

***

## Algorithm informations

Algorithm inputs:
- YOLOv5 model weights (such as 'yolov5s', 'yolov5l', 'yolov5x', etc..) 
- Re-Identification model weights (such as 'osnet_x0_25.pt', 'osnet_x0_75.pt', etc...)
- Source path (path of the video file that sould be process)

Algoritm outputs:
- Folder that contains video/videos processed

Formats accepted:
- IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
- VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

***

## Installation and usage

In a work environment with Python>=3.7 and torch installed, clone and install the official YOLOv5 repository using the following commands:
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

Then, clone and install this repository using the following commands:
```
git clone git@github.com:federicorossifr/eupilot-cini-mandown.git
cd eupilot-cini-mandown
pip install -r requirements.txt
```

Execute code:
```
python man_down_tracking.py
```

Sources:

    0, 1, 2, ...                     # webcam
    img.jpg                          # image
    vid.mp4                          # video
    path/                            # directory
    'path/*.jpg'                     # glob
    'https://youtu.be/Zgi9g1ksQHc'   # YouTube
    'rtsp://example.com/media.mp4'   # RTSP, RTMP, HTTP stream

Weights:

    yolov5s.pt                 # PyTorch
    yolov5s.onnx               # ONNX Runtime
    yolov5s.engine             # TensorRT

***
## Benchmark

![benchmark.png](benchmark.png)

***

#### Ultralytics YOLOv5 GitHub Official Repository
https://github.com/ultralytics/yolov5

***

#### DeepSORT GitHub Official Repository
https://github.com/nwojke/deep_sort