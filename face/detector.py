import cv2
import numpy as np
import config

class FaceDetector:
    def __init__(self):
        # the readNetFromCaffe method loads a pre-trained neural network in the caffe format
        self.model = cv2.dnn.readNetFromCaffe(config.CAFFE_PROTO, config.CAFFE_MODEL)

    def detect(self, frame):
        #dimensions
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123)) # scale, size, mean (values used in the normalization model)
        self.model.setInput(blob)
        detections = self.model.forward() # propagation
        # example:
        # [ batch, num_detections, detections, 7 ]
        # batch_id, class_id (not used), confidence, coordinates

        # processes detection
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf < 0.5:
                continue

            # converts normalized coordinates to pixels
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            # corrects the coordinates and stores
            faces.append((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
            # stores evert face as a tuple (x1, y1, x2, y2)
        return faces
