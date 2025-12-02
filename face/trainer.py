import os
import cv2
import config

class Trainer:
    def __init__(self, db, max_photos=300):
        self.db = db
        self.max_photos = max_photos

    def save_training_image(self, name, face_img, embedding):
        folder = os.path.join(config.DATA_DIR, name)
        os.makedirs(folder, exist_ok=True)

        # Count existing images
        existing = len(os.listdir(folder))

        # Stop if we have reached the limit
        if existing >= self.max_photos:
            return False   # <--- signal "not saved"

        # Save next image
        count = existing + 1
        path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(path, face_img)

        # Add embedding to DB
        self.db.add_face(name, embedding)

        return True  # <--- signal "saved successfully"
