# face/database.py

import os
import pickle
import numpy as np
import config

class FaceDatabase:
    def __init__(self):
        if os.path.exists(config.DB_FILE):
            with open(config.DB_FILE, "rb") as f:
                self.db = pickle.load(f)
        else:
            self.db = {"embeddings": [], "labels": []}

        self.update_arrays()

    def update_arrays(self):
        self.embeddings = np.array(self.db["embeddings"]) if self.db["embeddings"] else np.zeros((0,128))
        self.labels = np.array(self.db["labels"]) if self.db["labels"] else np.array([])

    def add_face(self, name, embedding):
        self.db["embeddings"].append(embedding)
        self.db["labels"].append(name)
        self.update_arrays()

    def save(self):
        with open(config.DB_FILE, "wb") as f:
            pickle.dump(self.db, f)
