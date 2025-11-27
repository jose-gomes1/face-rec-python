# face/trainer.py

import os
import cv2
import config

class Trainer:
    def __init__(self, db):
        self.db = db

    def save_training_image(self, name, face_img, embedding):
        folder = os.path.join(config.DATA_DIR, name)
        os.makedirs(folder, exist_ok=True)

        count = len(os.listdir(folder)) + 1
        path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(path, face_img)

        self.db.add_face(name, embedding)
