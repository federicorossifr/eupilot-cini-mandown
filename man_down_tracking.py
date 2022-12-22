''' Man Down Tracking'''

import os
import sys
import logging
from pathlib import Path
import argparse
import cv2
import numpy as np
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
from man_down_utils.general import non_max_suppression, time_sync, check_img_size, check_file, check_imshow, increment_path, select_device
from man_down_utils.loading import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from man_down_utils.visualization import Annotator, colors, colorstr
from man_down_utils.memorization import SaveData
from deep_sort.deep_sort import DeepSort
from man_down.man_down import ManDown

logging.getLogger().removeHandler(logging.getLogger().handlers[0])

# Parameters:
'''
Source:
0, 1, 2, ...                     # webcam
img.jpg                          # image
vid.mp4                          # video
path/                            # directory
'path/*.jpg'                     # glob
'https://youtu.be/Zgi9g1ksQHc'   # YouTube
'rtsp://example.com/media.mp4'   # RTSP, RTMP, HTTP stream
'''
# source = ROOT / 'sources/images/img7.jpg'
source = ROOT / 'sources/videos/vid7.mp4'
# source = 1
yolo_weights = WEIGHTS / 'yolov5x.pt'
reid_weights = WEIGHTS / 'osnet_x0_75_msmt17.pt'
yolo_classes_path = ROOT / 'configs/coco.yaml'
yolo_classes_to_detect = 0  # Convert to ['person'] ...
ratio_thres = 1.0  # w/h ratio threshold for man down filter
imgsz = (640, 640)
conf_thres = 0.25,  # confidence threshold (CONVERT TO INT AND PASS TO NMS)
iou_thres = 0.45,  # NMS IOU threshold (CONVERT TO INT AND PASS TO NMS)
max_det = 1000,  # maximum detections per image (CONVERT TO INT AND PASS TO NMS)
device = ''
view_img = False
save_txt = False  # save results to *.txt
save_conf = False
save_crops = False
classes_filter = 0  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False
nosave = False  # do not save images/videos
augment = False
visualize = False
update = False
exist_ok = False
project = ROOT / 'results'
name = 'test'
line_thickness = 3  # bounding box thickness (pixels)
hide_labels = False  # hide labels
hide_conf = False  # hide confidences
half = False
dnn = False
vid_stride = 1
evaluate = True
show_vid = False 
save_vid = True 
save_data = True

# Get Source Type:
source = str(source)
save_img = not nosave and not source.endswith('.txt')  # save inference images
is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
if is_url and is_file:
    source = check_file(source)  # download

# Directories:
save_dir = increment_path(Path(project) / name, exist_ok = exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents = True, exist_ok = True)  # make dir

# Load Object Detector (YOLOv5):
device = select_device(device)
# device = torch.device('cpu')
model = DetectMultiBackend(yolo_weights, device = device, dnn = dnn, data = yolo_classes_path, fp16 = half)
stride, names, pt, onnx, engine = model.stride, model.names, model.pt, model.onnx, model.engine
imgsz = check_img_size(imgsz, s = stride)  # check image size

# Load Man Down Filter:
man_down = ManDown(ratio_thres = ratio_thres)

# Load Deep Sort Algorithm:
deep_sort = DeepSort(reid_weights, device = device, fp16 = half)

# Load Data Saver:
data_saver = SaveData(save_dir, device)

# Dataloader:
bs = 1  # batch_size
if webcam:
    view_img = check_imshow(warn = True)
    dataset = LoadStreams(source, img_size = imgsz, stride = stride, auto = pt, vid_stride = vid_stride)
    bs = len(dataset)
else:
    dataset = LoadImages(source, img_size = imgsz, stride = stride, auto = pt, vid_stride = vid_stride)
vid_path, vid_writer = [None] * bs, [None] * bs

count = 0
data = []
def count_obj(box, w, h, id):
    global count, data
    center_coordinates = (int(box[0] + (box[2] - box[0])/2) , int(box[1]+(box[3] - box[1])/2))
    if int(box[1] + (box[3] - box[1])/2) > (h - 350):
        if  id not in data:
            count += 1
            data.append(id)

# Run Algorithm:
t_init = time_sync()
model.warmup(imgsz = (1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, dt = 0, [0.0, 0.0, 0.0, 0.0, 0.0]
for frame_idx, (path, img, img0s, vid_cap, s) in enumerate(dataset):

    # Pre-Process Image:
    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] = t2 - t1

    # Inference:
    visualize = increment_path(save_dir / Path(path).stem, mkdir = True) if visualize else False
    t3 = time_sync()
    pred = model(img, augment = augment, visualize = visualize)
    t4 = time_sync()
    dt[1] = t4 - t3

    # Apply NMS:
    t5 = time_sync()
    pred = non_max_suppression(pred, conf_thres = 0.25, iou_thres = 0.45, classes = yolo_classes_to_detect, max_det = 1000)
    t6 = time_sync()
    dt[2] = t6 - t5

    # Process Detections:
    for i, det in enumerate(pred):  # detections per image

        seen += 1
        if webcam:  # batch_size >= 1
            p, img0, _ = path[i], img0s[i].copy(), dataset.count
            s += f'webcam {i}: '
        else:
            p, img0, _ = path, img0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg, vid.mp4, ...
        s += '%gx%g ' % img.shape[2:]  # print string
        annotator = Annotator(img0, line_width = line_thickness, example = str(names))
        w, h = img0.shape[1], img0.shape[0]

        bboxes = det[:, :4]  # bounding boxes in xyxy form (torch tensor on device)
        scores = det[:, 4]  # bounding boxes scores (torch tensor on device)
        classesIDs = det[:, 5]  # bounding boxes classes IDs (torch tensor on device)

        # Pass Detections to Man Down Algorithm:
        t7 = time_sync()
        bboxes, scores, classesIDs = man_down.detection(img, img0, bboxes, scores, classesIDs)
        t8 = time_sync()
        dt[3] = t8 - t7

        # if det is not None and len(det):
        if scores is not None and len(scores):

            # Print results:
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)} "  # add to string

            # Pass Detections to Deep Sort Algorithm:
            t9 = time_sync()
            outputs = deep_sort.update(bboxes.cpu(), scores.cpu(), classesIDs.cpu(), img0)
            t10 = time_sync()
            dt[4] = t10 - t9

            # Draw bounding boxes for visualization:
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, scores)):

                    bboxes = output[0:4]  # bounding boxes in xyxy form
                    id = output[4]  # object number 
                    classesIDs = output[5]  # object class
                    count_obj(bboxes, w, h, id)
                    c = int(classesIDs)  # integer class
                    if c == 0:
                        md = 'man down'
                        label = f'{id} {md} {conf:.2f}'
                    else:
                        label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color = colors(c, True))

        else:
            deep_sort.increment_ages()

        img0 = annotator.result()

        # Show videos:
        if show_vid:
            cv2.imshow(str(p), img0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections):
        if save_vid:
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
    if save_data:
        speed_info = data_saver.get_speed_info(dt)
        if str(device) != 'cpu':
            GPU_info = data_saver.get_GPU_info()
            data_saver.save(speed_info, GPU_info)
        else:
            data_saver.save(speed_info)  # only speed info
    print(f"{s}")
    print(f'speed: {dt[0]*1000:.1f} ms for pre-process, {dt[1]*1000:.1f} ms for inference, {dt[2]*1000:.1f} ms for NMS, {dt[3]*1000:.1f} ms for Man Down, {dt[4]*1000:.1f} ms for Deep Sort')

t_final = time_sync()

# Compute FPS:
FPS = seen/(t_final - t_init)
print("FPS: ", FPS)

# Save results:
if save_data or save_vid:
    print('Results saved to %s' % save_path)