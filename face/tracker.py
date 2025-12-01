from scipy.spatial import distance
from collections import deque
import config

class FaceTracker:
    def __init__(self):
        self.next_id = 0
        self.faces = {}

    # receives the list of faces detected in this frame
    def assign(self, detections):
        assigned = [] # list initialization
        # example:
        # {
        #   "bbox": (x1,y1,x2,y2),
        #   "embedding": emb,
        #   "center": (cx, cy)
        # }

        # detection loop
        for det in detections:
            (x1,y1,x2,y2), emb, center = det["bbox"], det["embedding"], det["center"]

            # tries to associate with an existing ID
            face_id = None
            # calculates the euclidean distance between the center of the current face and the center of the previous face using ID fid
            for fid, data in self.faces.items():
                dist = distance.euclidean(center, data["center"])
                # if it's close enought, assumes as the same person
                if dist < config.TRACK_DIST_THRESHOLD:
                    face_id = fid
                    break

            if face_id is None:
                face_id = self.next_id
                self.next_id += 1
                self.faces[face_id] = {
                    "names": deque(maxlen=config.SMOOTH_FRAMES)
                }

            self.faces[face_id].update({"bbox":(x1,y1,x2,y2),"embedding":emb,"center":center})
            assigned.append(face_id)

        return assigned
