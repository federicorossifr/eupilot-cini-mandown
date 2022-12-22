'''Man Down Detector'''

import torch
from man_down_utils.general import xyxy2xywh, scale_boxes

class ManDown:

    def __init__(self, ratio_thres = 1.0):

        self.ratio_thres = ratio_thres

    def detection(self, img, img0, bboxes, scores, classesIDs):
        
        if len(scores) != 0:
            bboxes = scale_boxes(img.shape[2:], bboxes, img0.shape).round()
            bboxes = xyxy2xywh(bboxes)
            new_bboxes, new_scores, new_classesIDs = [], [], []

            for i in range(0, len(scores)):
                w = bboxes[i][2]
                h = bboxes[i][3]
                if w/h > self.ratio_thres:
                    new_bboxes.append(bboxes[i])
                    new_scores.append(scores[i])
                    new_classesIDs.append(classesIDs[i])

            if len(new_scores) != 0:
                new_bboxes = torch.stack(new_bboxes, dim = 0)
                new_scores = torch.stack(new_scores, dim = 0)
                new_classesIDs = torch.stack(new_classesIDs, dim = 0)
            
            else:
                new_bboxes = torch.empty(0)
                new_scores = torch.empty(0)
                new_classesIDs = torch.empty(0)
        
        else:
            new_bboxes = torch.empty(0)
            new_scores = torch.empty(0)
            new_classesIDs = torch.empty(0)

        return new_bboxes, new_scores, new_classesIDs