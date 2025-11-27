# config.py

import os

# -----------------------------
# MODEL PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAFFE_PROTO = os.path.join(BASE_DIR, "dataset/deploy.prototxt.txt")
CAFFE_MODEL = os.path.join(BASE_DIR, "dataset/res10_300x300_ssd_iter_140000.caffemodel")
FACENET_T7 = os.path.join(BASE_DIR, "dataset/nn4.small2.v1.t7")

DB_FILE = os.path.join(BASE_DIR, "dataset/face_db.pkl")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# -----------------------------
# THRESHOLDS
# -----------------------------
DIST_THRESHOLD = 0.6       # Face recognition threshold
TRACK_DIST_THRESHOLD = 50  # Pixel movement to keep tracking ID
SMOOTH_FRAMES = 10         # Recognition smoothing window

# -----------------------------
# PYGAME
# -----------------------------
WIDTH = 800
HEIGHT = 600
BG_COLOR = (30, 30, 30)
BUTTON_COLOR = (70,130,180)
BUTTON_HOVER = (100,160,210)
TEXT_COLOR = (255,255,255)
FONT_SIZE = 30
