# Man Down Tracking 🚀

"""
Run the DeepSORT algorithm

"""

import numpy as np
import torch

from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.reid.reid_multibackend import ReIDDetectMultiBackend

class DeepSORT(object):

    def __init__(self, reid_weights, parameters, device = 'cpu', fp16 = None):

        max_cosine_distance = parameters.get('MAX_DIST')
        max_iou_distance = parameters.get('MAX_IOU_DISTANCE')
        max_age = parameters.get('MAX_AGE')
        n_init = parameters.get('N_INIT')
        nn_budget = parameters.get('NN_BUDGET')
        lambda_weight = parameters.get('LAMBDA')

        self.model = ReIDDetectMultiBackend(weights = reid_weights, device = device, fp16 = fp16)
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance = max_iou_distance, max_age = max_age, n_init = n_init, lambda_weight = lambda_weight)

    def update(self, img0, det, use_yolo_preds = True):

        self.height, self.width = img0.shape[:2]
        xywhs = det[:, :4]
        scores = det[:, 4]
        classes = det[:, 5]

        # Get features with Re-Identification Model:
        features = self._get_features(img0, xywhs)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], score, features[i]) for i, score in enumerate(scores)]

        # Run on non-maximum supression:
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])

        # Predict and update tracks:
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # Output bbox identities:
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                dets = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(dets.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id]))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis = 0)

        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.

        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)

        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)

        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):

        x1, y1, x2, y2 = bbox_xyxy
        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)

        return t, l, w, h

    def _get_features(self, ori_img, bbox_xywh):

        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops).cpu().detach().numpy()
        else:
            features = np.array([])

        return features