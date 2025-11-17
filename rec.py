import cv2
import os
import numpy as np
from PIL import Image

DATASET_DIR = "dataset"

# Load DNN face detector once
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt, model)


def detect_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:  # adjust as needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # ---- CLAMP VALUES ----
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            width = x2 - x1
            height = y2 - y1

            # ---- IGNORE INVALID BOXES ----
            if width > 10 and height > 10:
                faces.append((x1, y1, width, height))

    return faces


def load_training_data(dataset_path=DATASET_DIR):
    face_samples = []
    ids = []
    names = {}
    current_id = 0

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

            # Load saved cropped face directly
            img = Image.open(img_path).convert("L")
            img_np = np.array(img, "uint8")

            # Skip empty/corrupted files
            if img_np is None or img_np.size == 0:
                continue

            face_samples.append(img_np)
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

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # DNN detection works in BGR directly
        detected_faces = detect_faces_dnn(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in detected_faces:
            if x < 0 or y < 0 or x+w > gray.shape[1] or y+h > gray.shape[0]:
                continue

            roi_gray = gray[y:y+h, x:x+w]                  # Crop face
            if roi_gray.size == 0:
                continue

            # ---- NEW: preprocessing ----
            roi_gray = cv2.resize(roi_gray, (200, 200))
            roi_gray = cv2.equalizeHist(roi_gray)

            # Predict
            id_, confidence = recognizer.predict(roi_gray)

            if confidence < 70:
                name = names.get(id_, "Unknown")
                color = (0, 255, 0)   # green
            else:
                name = "Unknown"
                color = (0, 0, 255)   # red

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
