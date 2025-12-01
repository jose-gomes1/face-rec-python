import os
import cv2
import config

class Trainer:
    def __init__(self, db):
        self.db = db

    # each person has their own folder within DATA_DIR
    def save_training_image(self, name, face_img, embedding):
        folder = os.path.join(config.DATA_DIR, name)
        os.makedirs(folder, exist_ok=True)

        # Counts how many images already exist in the person's folder and returns the next numbers: 1, 2, 3
        count = len(os.listdir(folder)) + 1
        path = os.path.join(folder, f"{count}.jpg")
        # saves the image of the face (already cut out) in this folder
        cv2.imwrite(path, face_img)

        self.db.add_face(name, embedding)
