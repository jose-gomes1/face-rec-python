# face/detector.py

import cv2
import numpy as np
import config

class FaceDetector:
    def __init__(self):
        self.model = cv2.dnn.readNetFromCaffe(config.CAFFE_PROTO, config.CAFFE_MODEL)

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123))
        self.model.setInput(blob)
        detections = self.model.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf < 0.5:
                continue

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            faces.append((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
        return faces
