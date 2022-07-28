import os
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from YOLOv5_PyTorch import YOLOv5_PyTorch
from utilities import LoadImages, LoadWebcam, Annotator, increment_path, colorstr, time_sync

class Algorithm:
    
    def __init__(self, model_weights, classes_dict_path, source_path, output_path, dir_name):

        self.source_path = source_path
        self.output_path = output_path
        self.dir_name = dir_name

        # YOLOv5 properties:
        yolov5_properties = {
            'model_weights': model_weights,
            'classes_dict_path': classes_dict_path,
            'classes_to_detect': ['person'],
            'nms_thresh': 0.4,
            'prob_thresh': 0.25,
        }

        # Initialize detector:
        self.detector = YOLOv5_PyTorch(properties = yolov5_properties)

        # Algorithm properties:
        self.num_pers_thresh = 10
        self.ratio_thresh = 1.0
        self.batch_size = 1
        
        setattr(self, 'ind', 0)  # detection id counter

    def processImage(self, image: np.ndarray, timestamp: float, keys: list = []) -> dict:

        det_out = self.detector.detect(image)
        annotator = Annotator(image)

        currDetections = []
        num_objects = len(det_out.get('scores'))  # get the number of relevant object detected

        for i in range(0, num_objects):
            box = det_out.get('boxes')[i]
            class_name = self.detector.class_mapping[det_out['classes'][i]]
            score = round(det_out.get('scores')[i], 2)

            # From xyxy to xywh:
            box = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
            x = int((box[0] + box[2]) / 2)  # x center
            y = int((box[1] + box[3]) / 2)  # y center
            w = int(box[2] - box[0])  # width
            h = int(box[3] - box[1])  # height

            # if w/h > self.ratio_thresh or len(det_out['boxes']) < self.num_pers_thresh:
            if w/h > self.ratio_thresh:
                # Override "resting" class with "man down"
                class_name = "man down"
                currDetections.append(
                    {
                        'id': self.ind,
                        'boundingbox': box.tolist(),
                        'class_name': class_name,
                        'score': score,
                        'timestamp': timestamp
                    }
                )
                self.ind += 1

        out_dict = {'currDetections': currDetections}

        num_man_down = len(out_dict.get('currDetections'))
        if num_man_down != 0:
            for idx in range(0, num_man_down):
                box = out_dict.get('currDetections')[idx].get('boundingbox')
                box = [box[0], box[1], box[2], box[3]]
                class_name = out_dict.get('currDetections')[idx].get('class_name')
                score = round(out_dict.get('currDetections')[idx].get('score'), 2)
                annotator.box_label(box, f'{class_name} {score}', color = (0, 0, 255))
                output_img = annotator.result()
        else:
            output_img = annotator.result()

        return output_img
        
    def run(self):

        exist_ok = False
        save_txt = False
        webcam = False

        # Directories
        save_dir = increment_path(Path(self.output_path) / self.dir_name, exist_ok = exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents = True, exist_ok = True)  # make dir

        source = str(self.source_path)
        dataset = LoadImages(source)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        count = 0
        for path, img0, vid_cap, s in dataset:
            timestamp = time.ctime(time.time())
            img_processed = self.processImage(img0, timestamp)
            p = Path(path)
            save_path = str(save_dir / p.name)  # im.jpg
            
            if dataset.mode == 'image':
                cv2.putText(img_processed, str(timestamp), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                cv2.imwrite(save_path, img_processed)
                print(f'{s}Done.')

            elif dataset.mode == 'video':
                if count == 0:
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, img0.shape[1], img0.shape[0]

                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    count = 1
                
                vid_writer.write(img_processed)
                print(f'{s}Done.')

        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('white', 'bold', save_dir)}{s}")

    def queryBatchSize(self) -> int:
        return self.batch_size