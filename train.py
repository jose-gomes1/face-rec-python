# recognize.py
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

DB_FILE = "face_mesh_model.pkl"
TOP_K = 3  # number of nearest neighbors to consider
THRESHOLD = 0.6  # embedding similarity threshold
SMOOTH_FRAMES = 5  # frames to persist identity

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# -------------------------
# Extract embedding from 468 landmarks
# -------------------------
def get_embedding(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    center = np.mean(pts[:, :2], axis=0)
    pts[:, :2] = (pts[:, :2] - center) / (np.linalg.norm(pts[:, :2] - center) + 1e-6)
    return pts.flatten()

# -------------------------
# Predict using top-k nearest neighbors + temporal smoothing
# -------------------------
class Recognizer:
    def __init__(self, db, top_k=TOP_K, threshold=THRESHOLD, smooth_frames=SMOOTH_FRAMES):
        self.db = db
        self.top_k = top_k
        self.threshold = threshold
        self.smooth_frames = smooth_frames
        self.last_ids = deque(maxlen=smooth_frames)

    def predict(self, embedding):
        if len(self.db["data"]) == 0:
            return "Unknown", 999

        # Compute distances to all embeddings
        distances = np.linalg.norm(np.array(self.db["data"]) - embedding, axis=1)
        nearest_idx = distances.argsort()[:self.top_k]
        nearest_labels = [self.db["labels"][i] for i in nearest_idx]
        nearest_dists = distances[nearest_idx]

        # Majority vote among top-k
        label_counts = {}
        for lbl, dist in zip(nearest_labels, nearest_dists):
            if dist < self.threshold:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        if not label_counts:
            return "Unknown", nearest_dists[0]

        best_label = max(label_counts.items(), key=lambda x: x[1])[0]
        best_dist = min([d for lbl, d in zip(nearest_labels, nearest_dists) if lbl == best_label])
        
        # Temporal smoothing
        self.last_ids.append(best_label)
        if len(self.last_ids) == self.smooth_frames:
            # if majority agrees in last few frames
            votes = {lbl: self.last_ids.count(lbl) for lbl in set(self.last_ids)}
            if max(votes.values()) / self.smooth_frames > 0.6:  # 60% agreement
                return best_label, best_dist
            else:
                return "Unknown", best_dist
        else:
            return best_label, best_dist

# -------------------------
# Main loop
# -------------------------
def main():
    print("[INFO] Loading database...")
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)

    recognizer = Recognizer(db)

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    # Full 468-point embedding
                    embedding = get_embedding(fl.landmark)

                    # Predict
                    name, dist = recognizer.predict(embedding)

                    # Compute bounding box
                    xs = [int(lm.x * w) for lm in fl.landmark]
                    ys = [int(lm.y * h) for lm in fl.landmark]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} ({dist:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Draw mesh (light gray)
                    for lm in fl.landmark:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (px, py), 1, (200, 200, 200), -1)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
