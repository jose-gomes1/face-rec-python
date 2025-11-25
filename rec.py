import cv2
import numpy as np
import pickle
from collections import deque, Counter
from scipy.spatial import distance

# ----------------------------
# Paths
# ----------------------------
CAFFE_PROTO = "deploy.prototxt.txt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACENET_T7 = "nn4.small2.v1.t7"
DB_FILE = "face_db.pkl"

# Config
SMOOTH_FRAMES = 10
DIST_THRESHOLD = 0.6       # Distance threshold for unknown
TRACK_DIST_THRESHOLD = 50  # pixels max movement to keep same ID

# ----------------------------
# Load models and database
# ----------------------------
face_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)
embedder = cv2.dnn.readNetFromTorch(FACENET_T7)

with open(DB_FILE, "rb") as f:
    db = pickle.load(f)

db_embeddings = np.array(db["embeddings"])
db_labels = np.array(db["labels"])

# ----------------------------
# Tracking buffers
# ----------------------------
next_face_id = 0
faces = {}  # face_id -> {'bbox':(x1,y1,x2,y2), 'embedding':vec, 'names':deque, 'center':(cx,cy)}

# ----------------------------
# Video capture
# ----------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    current_faces = []

    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf < 0.5:
            continue

        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)

        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (96,96))
        face_blob = cv2.dnn.blobFromImage(face_resized, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        vec = embedder.forward().flatten()

        cx, cy = (x1+x2)//2, (y1+y2)//2
        current_faces.append({'bbox':(x1,y1,x2,y2),'embedding':vec,'center':(cx,cy)})

    # ----------------------------
    # Assign IDs using centroid tracking
    # ----------------------------
    assigned_ids = []
    for cf in current_faces:
        min_dist = float('inf')
        assigned_id = None
        for fid, data in faces.items():
            prev_cx, prev_cy = data['center']
            dist_c = distance.euclidean(cf['center'], (prev_cx, prev_cy))
            if dist_c < TRACK_DIST_THRESHOLD:
                assigned_id = fid
                break

        if assigned_id is None:
            assigned_id = next_face_id
            next_face_id += 1
            faces[assigned_id] = {'bbox':cf['bbox'],
                                  'embedding':cf['embedding'],
                                  'names':deque(maxlen=SMOOTH_FRAMES),
                                  'center':cf['center']}
        else:
            faces[assigned_id]['bbox'] = cf['bbox']
            faces[assigned_id]['embedding'] = cf['embedding']
            faces[assigned_id]['center'] = cf['center']

        assigned_ids.append(assigned_id)

    # ----------------------------
    # Recognition + smoothing
    # ----------------------------
    for fid in assigned_ids:
        vec = faces[fid]['embedding']
        # Compare to DB
        dists = np.linalg.norm(db_embeddings - vec, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        name = db_labels[min_idx] if min_dist < DIST_THRESHOLD else "Unknown"

        faces[fid]['names'].append(name)
        smoothed_name = Counter(faces[fid]['names']).most_common(1)[0][0]

        x1,y1,x2,y2 = faces[fid]['bbox']
        color = (0,255,0) if smoothed_name!="Unknown" else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{smoothed_name} ({min_dist:.2f})",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("Face Recognition with Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
