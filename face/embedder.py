# face/embedder.py

import cv2
import config

class FaceEmbedder:
    def __init__(self):
        self.model = cv2.dnn.readNetFromTorch(config.FACENET_T7)

    def get_embedding(self, face_img):
        resized = cv2.resize(face_img, (96,96))
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, (96,96), (0,0,0), swapRB=True)
        self.model.setInput(blob)
        return self.model.forward().flatten()
