# EuPilot CINI - YOLOv5 Man Down Tracking

This repository contains a configurable man down tracker. The detections generated by YOLOv5, one of the most popular object detection algorithm, are passed to the DeepSORT algorithm that implements tracking and counting tasks.

***

## Algorithm informations

Algorithm inputs:
- YOLOv5 model weights (such as 'yolov5s.pt', 'yolov5l.pt', 'yolov5x.pt', etc..) 
- Re-Identification model weights (such as 'osnet_x1_0_market1501.pt', 'osnet_x0_75_market1501.pt', 'osnet_x0_25_market1501.pt', etc...)
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
Finally, install requirements using the following commands:
```
cd ..
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

All the tests are made using YOLOv5x model and OSNet x1.0 model.
   
platform 1: **ARM Cortex-A72**  
platform 2: **ARM Neoverse N1**  
platform 3: **Fujitsu A64FX** (ARMv8-A based)   
platform 4: **Intel i7-10750H**     
platform 5: **Intel i7-10750H + NVIDIA GeForce GTX 1650 Ti**    
platform 6: **Intel Xeon**     
platform 7: **Intel Xeon + NVIDIA Tesla T4**  

| Platform | FPS | YOLO Inference Speed<br>(ms) | Man Down Classifier Speed<br>(ms) | DeepSORT Speed<br>(ms) | CPU Temperature<br>(°C) | CPU Power Consumption<br>(W) | GPU Temperature<br>(°C) | GPU Power Consumption<br>(W) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.1 | **7632** | 1.2 | **1032** | 81.9 | - | - | - |
| 2 | 0.2 | **878** | 0.5 | **3794** | 51.6 | - | - | - |
| 3 | 0.4 | **1221** | 1.1 | **1233** | - | - | - | - |
| 4 | 1.0 | **794** | 0.3 | **200** | 95 | - | - | - |
| 5 | 6.5 | **82.9** | 0.3 | **37.9** | - | - | 74.8 | 38.5 |
| 6 | 1.8 | **335** | 0.3 | **197** | - | - | - | - |
| 7 | 11.6 | **33.9** | 0.3 | **28.3** | - | - | 49.4 | 57 |

<p align = "center"><img width="600" src="yolo_inference_speed.png"></p>

<p align = "center"><img width="600" src="deep_sort_speed.png"></p>

***

#### Ultralytics YOLOv5 GitHub Official Repository
https://github.com/ultralytics/yolov5

***

#### DeepSORT GitHub Official Repository
https://github.com/nwojke/deep_sort