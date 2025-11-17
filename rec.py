import cv2
import os
import numpy as np
from PIL import Image

DATASET_DIR = "dataset"


def load_training_data(dataset_path=DATASET_DIR):
    face_samples = []
    ids = []
    names = {}
    current_id = 0

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Loop through persons
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"[INFO] Loading images for: {person_name}")
        names[current_id] = person_name

        for img_file in os.listdir(person_folder):
            if not img_file.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(person_folder, img_file)
            img = Image.open(img_path).convert("L")  # grayscale
            img_np = np.array(img, "uint8")

            faces = face_cascade.detectMultiScale(img_np, 1.3, 5)

            # Add all detected faces for training
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(current_id)

        current_id += 1

    return face_samples, ids, names


def recognize():
    print("[INFO] Training face recognizer from dataset...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids, names = load_training_data()
    recognizer.train(faces, np.array(ids))

    print("[INFO] Training complete. Starting camera...")

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            roi_gray = gray[y:y+h, x:x+w]

            id_, confidence = recognizer.predict(roi_gray)

            # Lower confidence = better match
            if confidence < 70:
                name = names.get(id_, "Unknown")
                color = (0, 255, 0)   # green
            else:
                name = "Unknown"
                color = (0, 0, 255)   # red

            # Draw box & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({round(confidence,1)})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
