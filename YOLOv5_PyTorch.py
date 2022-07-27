import json
import numpy as np
import torch
from utilities import time_sync

class YOLOv5_PyTorch:

    def __init__(self, properties = {}):

        self.properties = properties
        self.model_weights = self.properties.get('model_weights')
        self.classes_dict_path = self.properties.get('classes_dict_path')
        self.classes_to_detect = self.properties.get('classes_to_detect')
        self.nms_thresh = self.properties.get('nms_thresh')
        self.prob_thresh = self.properties.get('prob_thresh')

        print(f"""YOLOv5 PyTorch model properties:
        Model Weights: {self.model_weights}
        Classes to Detect: {self.classes_to_detect}
        Non Max Suppression Threshold: {self.nms_thresh}
        Probability Threshold: {self.prob_thresh}""")

        self.gpu_id = 0
        self.model = torch.hub.load('ultralytics/yolov5', self.model_weights)
        
        # YOLO classes mapping:
        with open(self.classes_dict_path, 'r') as f:
            self.class_names = json.load(f)
        self.class_mapping = {int(k): v for k, v in self.class_names.items()}

    def detect(self, np_img):

        # Inference:
        t1 = time_sync()
        output = self.model(np_img)
        t2 = time_sync()
        dt1 = t2 - t1  # inference speed

        # Post-Process:
        result_boxes, result_scores, result_classid = self.post_process(output)  # image post-processing
        out_boxes, out_classes, out_scores = [], [], []
        if result_boxes.shape[0] != 0:
            for det_idx in range(result_boxes.shape[0]):
                if self.class_names[str(int(result_classid[det_idx].item()))] in self.classes_to_detect:
                    out_boxes.append(result_boxes[det_idx])  # output xyxy boxes
                    out_classes.append(int(result_classid[det_idx].item()))  # output classes array
                    out_scores.append(result_scores[det_idx].item())  # output scores array
        t3 = time_sync()
        dt2 = t3 - t2  # post-process speed

        return {"boxes": out_boxes, "scores": out_scores, "classes": out_classes}

    def post_process(self, output):
        # Post-process model output
        pred = output.pandas().xyxy[0]
        pred = pred.to_numpy()  # from Pandas to NumPy
        pred = np.delete(pred, 6, 1)  # Delete class names
        pred = pred[None]
        pred = pred.astype(np.float32)
        pred = torch.from_numpy(pred).to(self.gpu_id)  # output image to a torch Tensor (necessary for NMS)
        result_boxes = pred[0][:, 0:4]  # get bounding boxes
        result_scores = pred[0][:, 4]  # get scores
        result_classid = pred[0][:, 5]  # get classes ID

        return result_boxes, result_scores, result_classid

    def get_classes_to_detect(self):
        # Get a list of the classes to detect
        return list(self.class_names.values())