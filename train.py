import cv2
import os
import pickle
import numpy as np

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = "dataset"       # dataset/<person_name>/images
DB_FILE = "face_db.pkl"
CAFFE_PROTO = "deploy.prototxt.txt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACENET_T7 = "nn4.small2.v1.t7"  # Torch model

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Load models
# ----------------------------
face_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)
embedder = cv2.dnn.readNetFromTorch(FACENET_T7)

# ----------------------------
# Load or create DB
# ----------------------------
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
else:
    db = {"embeddings": [], "labels": []}

# ----------------------------
# Get person name
# ----------------------------
person_name = input("Enter person name: ").strip()
person_dir = os.path.join(DATA_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)
count = len(os.listdir(person_dir))

print(f"[INFO] Capturing faces for {person_name}. Press 'q' to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf < 0.5:
            continue

        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)

        face_img = frame[y1:y2, x1:x2]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Resize face for Facenet
        face_resized = cv2.resize(face_img, (96,96))
        face_blob = cv2.dnn.blobFromImage(face_resized, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        vec = embedder.forward().flatten()

        # Save embedding
        db["embeddings"].append(vec)
        db["labels"].append(person_name)

        # After saving a captured face and embedding
        count += 1
        img_path = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_path, face_img)

        # Display on frame
        cv2.putText(frame, f"Images captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        cv2.imshow("Capture", frame)

        # Also print to console
        print(f"[INFO] Saved {img_path} - total images: {count}")


    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save DB
with open(DB_FILE, "wb") as f:
    pickle.dump(db, f)

print(f"[INFO] Training complete. Total images for {person_name}: {count}")
