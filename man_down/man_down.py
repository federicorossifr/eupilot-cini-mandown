# Man Down Tracking 🚀

import numpy as np
from tools.general import xyxy2xywh, scale_coords

class ManDown:

    def __init__(self, ratio_thres, fp16 = False):

        self.ratio_thres = ratio_thres
        self.fp16 = fp16

    def process_image(self, img, img0, det):

        xyxys = det[:, :4].detach().numpy()  # boundissh.hca.bsc.esng boxes in xyxy form (NumPy array on CPU)
        scores = det[:, 4].detach().numpy()  # scores (NumPy array on CPU)
        classes = det[:, 5].detach().numpy()  # classes IDs (NumPy array on CPU)
        xyxys = scale_coords(img.shape[2:], xyxys, img0.shape).round()
        # xyxys = scale_boxes(img.shape[2:], xyxys, img0.shape).round()  # rescale boxes to the original image shape
        xywhs = xyxy2xywh(xyxys)  # convert bounding boxes from xyxy to xywh form
        output = []

        for i in range(0, len(det)):
            w = xywhs[i][2]
            h = xywhs[i][3]
            if w/h > self.ratio_thres:
                output.append([xywhs[i][0], xywhs[i][1], xywhs[i][2], xywhs[i][3], scores[i], classes[i]])

        output = np.array(output)
        output = np.ndarray.astype(output, dtype = np.half) if self.fp16 else np.ndarray.astype(output, dtype = np.float32)

        return output if len(output) else np.empty(0)