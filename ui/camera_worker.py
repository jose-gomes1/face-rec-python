# ui/camera_worker.py

import threading
import cv2
import numpy as np
import config

class CameraWorker(threading.Thread):
    def __init__(self, detector, embedder, db, recognizer, trainer, callback, train_name=None):
        super().__init__(daemon=True)
        self.detector = detector
        self.embedder = embedder
        self.db = db
        self.recognizer = recognizer
        self.trainer = trainer
        self.callback = callback
        self.train_name = train_name
        self.stop_flag = False
        self.result_text = ""

    def run(self):
        cap = cv2.VideoCapture(0)
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            faces = self.detector.detect(frame)

            for (x1,y1,x2,y2) in faces:
                face_img = frame[y1:y2, x1:x2]
                emb = self.embedder.get_embedding(face_img)

                if self.train_name:
                    self.trainer.save_training_image(self.train_name, face_img, emb)
                    name = self.train_name
                    self.result_text = f"Treinando {name}"
                else:
                    name, dist = self.recognizer.recognize(emb)
                    if name != "Unknown":
                        self.result_text = f"Reconhecido como: {name} ({dist:.2f})"
                    else:
                        self.result_text = "Não reconhecido"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if name!="Unknown" else (0,0,255), 2)
                cv2.putText(frame, name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if name!="Unknown" else (0,0,255), 2)

            # Pass both frame and result_text to the callback
            self.callback(frame, self.result_text)

        cap.release()
        if self.train_name:
            self.db.save()
