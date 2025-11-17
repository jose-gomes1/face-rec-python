import cv2
import os
import time
import numpy as np

# DNN model files (make sure they are in the same folder as this script)
PROTOTXT = "deploy.prototxt.txt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# Load the DNN face detector once
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


def detect_faces_dnn(frame):
    """Detect faces (front/side/tilted) using OpenCV DNN."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp bounding box to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            bw = x2 - x1
            bh = y2 - y1

            if bw > 20 and bh > 20:
                faces.append((x1, y1, bw, bh))

    return faces


def create_or_update_person_dataset(base_path="dataset"):
    person_name = input("Enter the name of the person: ").strip()

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    person_dir = os.path.join(base_path, person_name)

    if os.path.exists(person_dir):
        print(f"[INFO] Updating existing dataset for '{person_name}'.")
    else:
        print(f"[INFO] Creating new dataset for '{person_name}'.")
        os.makedirs(person_dir)

    return person_dir


def capture_images(person_dir, required_count=500):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Camera not found.")
        return

    # Determine starting count to append images
    existing_files = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    if existing_files:
        counts = [int(os.path.splitext(f)[0]) for f in existing_files]
        count = max(counts)
    else:
        count = 0

    print(f"\n[INFO] Starting image capture from {count+1}...")
    print("[INFO] Move your head in different angles.")
    time.sleep(1)

    while count < required_count + count:
        ret, frame = cam.read()
        if not ret:
            break

        faces = detect_faces_dnn(frame)

        for (x, y, w, h) in faces:
            count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_img = gray[y:y+h, x:x+w]

            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)

            # show progress
            cv2.putText(frame, f"Images: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Training Camera - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\n[INFO] Captured {count} images in total.")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_dir = create_or_update_person_dataset()
    capture_images(person_dir)
