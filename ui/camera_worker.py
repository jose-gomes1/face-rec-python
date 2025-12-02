import threading
import os
import config
import cv2

class CameraWorker(threading.Thread):
    def __init__(self, detector, embedder, db, recognizer, trainer, callback, train_name=None):
        super().__init__(daemon=True) # ensures that the thread terminates automatically when the main program closes
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
        cap = cv2.VideoCapture(0) # opens the camera
        while not self.stop_flag:
            ret, frame = cap.read() # reads every camera frame
            if not ret: continue

            frame = cv2.flip(frame, 1) # flips the frame (common in webcams)
            faces = self.detector.detect(frame) # decects the face in the frame

            for (x1,y1,x2,y2) in faces: # for every detected face
                face_img = frame[y1:y2, x1:x2] # cuts the face region
                emb = self.embedder.get_embedding(face_img) # generates the face embedding

                # if train_name is set, saves the face in the database (training)
                if self.train_name:
                    saved = self.trainer.save_training_image(self.train_name, face_img, emb)
                    if saved:
                        name = self.train_name
                        total = self.trainer.max_photos
                        folder = os.path.join(config.DATA_DIR, name)
                        current = len(os.listdir(folder))
                        self.result_text = f"Treinando {name}: {current}/{total}"
                    else:
                        self.result_text = f"Treinamento concluído ({self.trainer.max_photos} fotos)"
                        self.stop_flag = True
                else:
                    name, dist = self.recognizer.recognize(emb)
                    if name != "Unknown":
                        self.result_text = f"Reconhecido como: {name} ({dist:.2f})"
                    else:
                        self.result_text = "Não reconhecido"

                print(self.result_text)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if name!="Unknown" else (0,0,255), 2)
                cv2.putText(frame, name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if name!="Unknown" else (0,0,255), 2)

            # pass both frame and result_text to the callback
            self.callback(frame, self.result_text)

        cap.release()
        if self.train_name:
            self.db.save()
