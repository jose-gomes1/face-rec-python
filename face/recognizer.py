import numpy as np
import config

class Recognizer:
    def __init__(self, face_db):
        self.db = face_db

    def recognize(self, embedding):
        # If there are no saved embeddings, there's nothing to compare it to, so returns Unknown
        if self.db.embeddings.shape[0] == 0:
            return "Unknown", 999

        # calculates the euclidean distance between the given embedding and all embeddings in the database
        dists = np.linalg.norm(self.db.embeddings - embedding, axis=1)
        # finds the index for the smallest distance
        idx = np.argmin(dists)
        d = dists[idx]

        # checks if the distance is small enough
        if d < config.DIST_THRESHOLD:
            return self.db.labels[idx], d # recognizes the person and returns them
        return "Unknown", d
