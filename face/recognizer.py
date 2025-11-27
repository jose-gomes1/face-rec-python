# face/recognizer.py

import numpy as np
import config

class Recognizer:
    def __init__(self, face_db):
        self.db = face_db

    def recognize(self, embedding):
        if self.db.embeddings.shape[0] == 0:
            return "Unknown", 999

        dists = np.linalg.norm(self.db.embeddings - embedding, axis=1)
        idx = np.argmin(dists)
        d = dists[idx]

        if d < config.DIST_THRESHOLD:
            return self.db.labels[idx], d
        return "Unknown", d
