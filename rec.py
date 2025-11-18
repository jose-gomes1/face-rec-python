import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

DB_FILE = "face_mesh_model.pkl"
THRESHOLD = 0.65       # cosine similarity threshold
ROLLING_FRAMES = 5      # number of frames to average
TOP_K = 3               # top matches to display

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# ----------------------
# Normalize full 468-point embedding
# ----------------------
def get_embedding(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    center = np.mean(pts[:, :2], axis=0)
    pts[:, :2] = (pts[:, :2] - center) / (np.linalg.norm(pts[:, :2] - center) + 1e-6)
    return pts.flatten()

# ----------------------
# Cosine similarity
# ----------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

# ----------------------
# Recognize a single embedding with top-k
# ----------------------
def recognize_embedding(embedding, db):
    sims = [(cosine_similarity(embedding, vec), label) for vec, label in zip(db["data"], db["labels"])]
    sims.sort(reverse=True)  # highest similarity first
    top_sims = sims[:TOP_K]
    best_sim, best_label = top_sims[0]
    if best_sim < THRESHOLD:
        best_label = "Unknown"
    return best_label, best_sim, top_sims

# ----------------------
# Main
# ----------------------
def main():
    print("[INFO] Loading database...")
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)

    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)

    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Rolling embeddings per detected face (dict keyed by bbox center)
    face_memory = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        current_faces = []
        if (det_results := face_detection.process(frame_rgb)).detections:
            for det in det_results.detections:
                bbox_rel = det.location_data.relative_bounding_box
                x1 = max(int(bbox_rel.xmin * w), 0)
                y1 = max(int(bbox_rel.ymin * h), 0)
                x2 = min(int((bbox_rel.xmin + bbox_rel.width) * w), w-1)
                y2 = min(int((bbox_rel.ymin + bbox_rel.height) * h), h-1)

                face_crop = frame_rgb[y1:y2, x1:x2]
                mesh_results = face_mesh.process(face_crop)

                if mesh_results.multi_face_landmarks:
                    for fl in mesh_results.multi_face_landmarks:
                        embedding = get_embedding(fl.landmark)
                        # use bbox center as key
                        cx, cy = (x1 + x2)//2, (y1 + y2)//2
                        key = (cx, cy)

                        if key not in face_memory:
                            face_memory[key] = deque(maxlen=ROLLING_FRAMES)
                        face_memory[key].append(embedding)

                        # ----------------------
                        # Average embeddings over last frames
                        avg_emb = np.mean(np.array(face_memory[key]), axis=0)
                        name, score, top_matches = recognize_embedding(avg_emb, db)

                        # Draw polygon mesh in light gray
                        for lm in fl.landmark:
                            px = int(lm.x * (x2 - x1)) + x1
                            py = int(lm.y * (y2 - y1)) + y1
                            cv2.circle(frame, (px, py), 1, (200, 200, 200), -1)

                        # Draw bounding box and label
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Label with top-k suggestions if unknown
                        label_text = f"{name} ({score:.2f})"
                        if name == "Unknown":
                            suggestions = ", ".join([f"{l} ({s:.2f})" for s, l in top_matches])
                            label_text += f" | Top: {suggestions}"
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        current_faces.append(key)

        # Clean memory for disappeared faces
        keys_to_delete = [k for k in face_memory if k not in current_faces]
        for k in keys_to_delete:
            del face_memory[k]

        cv2.imshow("FaceMesh Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
