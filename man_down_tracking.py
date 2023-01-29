"""
Man Down Tracking

Run man down tracking algorithm on videos, YouTube, webcam, streams, etc.

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
    yolov5s.onnx               # ONNX
    yolov5s.engine             # TensorRT

"""

import os
import sys
import platform
from pathlib import Path
import numpy as np
import cv2
import torch

FILE = Path(__file__).resolve() # current file path (man_down_tracking.py path)
ROOT = FILE.parents[0]  # current file root path
WEIGHTS = ROOT / 'weights'  # weights folder path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 to PATH
if str(ROOT / 'deep_sort') not in sys.path:
    sys.path.append(str(ROOT / 'deep_sort'))  # add deep_sort to PATH
if str(ROOT / 'man_down') not in sys.path:
    sys.path.append(str(ROOT / 'man_down'))  # add man_down to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from deep_sort.deep_sort import DeepSORT
from man_down.man_down import ManDown
from tools.general import (Annotator, colors, time_sync, check_img_size, check_file, check_imshow, 
                            increment_path, select_device, non_max_suppression)
from tools.load import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS, load_yaml
from tools.save import SaveInfo

# Parameters:
source = ROOT / 'data/videos/test1.mp4'  # file/dir/URL/glob/screen/0(webcam)
yolo_weights = WEIGHTS / 'yolov5x.onnx'  # YOLOv5 model path
reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # ReID model path
deep_sort_params_path = ROOT / 'data/deep_sort.yaml'  # dataset.yaml path
classes_path = ROOT / 'data/coco.yaml'  # dataset.yaml path
classes_to_detect = 0  # filter by class: 0 or 0, 1, 2, 3
mandown_thres = 1.0  # man down classifier aspect ratio threshold
imgsz = (640, 640)  # inference image size (height, width)
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
project = ROOT / 'results'  # save results to project/name
name = 'test'  # save results to project/name
line_thickness = 3  # bounding box thickness (pixels)
half = False  # use FP16 half-precision inference
vid_stride = 1  # video frame-rate stride
view_img = False  # show results
save_img = False  # save images
save_txt = True  # save data to *.txt
exist_ok = False  # existing project/name ok, do not increment
augment = False  # augmented inference
visualize = False  # visualize features

# Get Source Type:
source = str(source)
is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
if is_url and is_file:
    source = check_file(source)  # download

# Directories:
save_dir = increment_path(Path(project) / name, exist_ok = exist_ok)  # increment run
save_dir.mkdir(parents = True, exist_ok = True)  # make dir

# Load Object Detector:
device = select_device(device)
model = DetectMultiBackend(yolo_weights, device = device, data = classes_path, fp16 = half)
stride, names = model.stride, model.names
pt, onnx, engine = model.pt, model.onnx, model.engine
imgsz = check_img_size(imgsz, s = stride)  # check image size

# Load Man Down Classifier:
man_down = ManDown(ratio_thres = mandown_thres, fp16 = half)

# Load Object Tracker:
deep_sort_params = load_yaml(deep_sort_params_path).get('parameters')
deep_sort = DeepSORT(reid_weights, parameters = deep_sort_params, device = device, fp16 = half)

# Load Logger:
data_logger = SaveInfo(save_dir, device)

# Dataloader:
bs = 1  # batch_size
if webcam:
    view_img = check_imshow(warn = True)
    dataset = LoadStreams(source, img_size = imgsz, stride = stride, auto = pt, vid_stride = vid_stride)
    bs = len(dataset)
else:
    dataset = LoadImages(source, img_size = imgsz, stride = stride, auto = pt, vid_stride = vid_stride)
vid_path, vid_writer = [None] * bs, [None] * bs

# Run Algorithm:
model.warmup(imgsz = (1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], [0.0, 0.0, 0.0, 0.0, 0.0]
t_init = time_sync()
for path, img, img0s, vid_cap, s in dataset:
    # Pre-Process Image:
    t1 = time_sync()
    img = np.ndarray.astype(img, dtype = np.half) if half else np.ndarray.astype(img, dtype = np.float32)  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] = t2 - t1  # pre-process speed

    # Inference:
    t3 = time_sync()
    visualize = increment_path(save_dir / Path(path).stem, mkdir = True) if visualize else False
    img = torch.from_numpy(img).to(model.device)
    pred = model(img, augment = augment, visualize = visualize)
    t4 = time_sync()
    dt[1] = t4 - t3  # inference speed

    # Apply NMS:
    t5 = time_sync()
    pred = pred[0].cpu() if pt else pred.cpu()
    pred = non_max_suppression(pred, conf_thres = 0.25, iou_thres = 0.45, classes = classes_to_detect, max_det = 1000)
    t6 = time_sync()
    dt[2] = t6 - t5  # post-process speed

    # Process Predictions:
    for i, det in enumerate(pred):  # per image
        seen += 1
        if webcam:  # batch_size >= 1
            p, img0, frame = path[i], img0s[i].copy(), dataset.count
            s += f'{i}: '
        else:
            p, img0, frame = path, img0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg, vid.mp4, ...
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        annotator = Annotator(img0, line_width = line_thickness, example = str(names))

        if len(det):
            
            # Pass Detections to the Man Down Classifier:
            t7 = time_sync()
            md_det = man_down.classify(img, img0, det)
            t8 = time_sync()
            dt[3] = t8 - t7  # man down classifier speed

            if len(md_det):

                # Print results
                for c in np.unique(md_det[:, 5]):
                    n = int(np.sum(md_det[:, 5] == c))  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Pass Detections to the DeepSORT algorithm:
                t9 = time_sync()
                outputs = deep_sort.update(img0, md_det)
                t10 = time_sync()
                dt[4] = t10 - t9  # deepSORT speed

                # Draw bounding boxes for visualization:
                if len(outputs):
                    scores = md_det[:, 4]
                    for j, (output, score) in enumerate(zip(outputs, scores)):
                        xyxy = output[0:4]  # bounding box in xyxy form
                        id = int(output[4])  # object number
                        class_id = output[5]  # object class
                        c = int(class_id)  # integer class
                        if c == 0:
                            md = 'man down'
                            label = f'{id} {md} {score:.2f}'
                        else:
                            label = f'{id} {names[c]} {score:.2f}'
                        annotator.box_label(xyxy, label, color = colors(c, True))
        else:
            pass

        img0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), img0.shape[1], img0.shape[0])
            cv2.imshow(str(p), img0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections):
        if save_img:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, img0.shape[1], img0.shape[0]

                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(img0)

    # Save data:
    if save_txt:
        speed_info = data_logger.get_speed_informations(dt)  # get speed informations
        CPU_info = data_logger.get_CPU_informations()  # get CPU informations
        if str(device) == 'cpu':
            data_logger.save(speed_info, CPU_info)
        else:
            GPU_info = data_logger.get_GPU_informations()  # get GPU informations (if available)
            data_logger.save(speed_info, CPU_info, GPU_info)

    print(f"{s}")
    print(f'speed: {dt[0]*1000:.1f} ms for pre-process, {dt[1]*1000:.1f} ms for inference, {dt[2]*1000:.1f} ms for post-process, {dt[3]*1000:.1f} ms for man down, {dt[4]*1000:.1f} ms for DeepSORT')

t_final = time_sync()

# Compute FPS:
FPS = seen/(t_final - t_init)
print("FPS: ", FPS)

# Save results:
if save_txt or save_img:
    print('Results saved to %s' % save_path)