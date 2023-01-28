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

In a work environment with **Python>=3.7** and **torch>=1.7** installed, clone this repository using the following commands:
```
git clone https://github.com/federicorossifr/eupilot-cini-mandown.git
```
Then, clone and install the official YOLOv5 repository using the following commands:
```
cd eupilot-cini-mandown
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
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

platform 1: **CPU ARM Cortex-A72**     
platform 2: **CPU Intel i7-10750H**     
platform 3: **CPU Intel i7-10750H + GPU NVIDIA GeForce GTX 1650 Ti + TensorRT**    
platform 4: **CPU Intel Xeon**     
platform 5: **CPU Intel Xeon + GPU NVIDIA Tesla T4 + TensorRT**    

| Platform | FPS | YOLOv5x Inference Speed<br>(ms) | Man Down Classifier Speed<br>(ms) | DeepSORT Speed<br>(ms) | CPU Temperature<br>(°C) | GPU Temperature<br>(°C) | GPU Power Consumption<br>(W) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | - | **7955.1** | - | - | - | - | - | - | - |
| 2 | - | **860.9** | - | - | - | - | - | - | - |
| 3 | - | **86.0** | - | - | - | - | - | - | - |
| 4 | - | **341.6** | - | - | - | - | - | - | - |
| 5 | - | **37.8** | - | - | - | - | - | - | - |

<p align = "center"><img width="600" src="benchmark.png"></p>

***

#### Ultralytics YOLOv5 GitHub Official Repository
https://github.com/ultralytics/yolov5

***

#### DeepSORT GitHub Official Repository
https://github.com/nwojke/deep_sort