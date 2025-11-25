import pygame
import threading
import cv2
import numpy as np
import pickle
import os
from collections import deque, Counter
from scipy.spatial import distance

# ----------------------------
# Paths
# ----------------------------
CAFFE_PROTO = "deploy.prototxt.txt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACENET_T7 = "nn4.small2.v1.t7"
DB_FILE = "face_db.pkl"
DATA_DIR = "dataset"

# ----------------------------
# Config
# ----------------------------
WIDTH, HEIGHT = 800, 600
BG_COLOR = (30, 30, 30)
BUTTON_COLOR = (70,130,180)
BUTTON_HOVER = (100,160,210)
TEXT_COLOR = (255,255,255)
FONT_SIZE = 30
SMOOTH_FRAMES = 10
DIST_THRESHOLD = 0.6
TRACK_DIST_THRESHOLD = 50

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Face Recognition App")
font = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()

# ----------------------------
# Load models and DB
# ----------------------------
face_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)
embedder = cv2.dnn.readNetFromTorch(FACENET_T7)

if os.path.exists(DB_FILE):
    with open(DB_FILE,"rb") as f:
        db = pickle.load(f)
else:
    db = {"embeddings": [], "labels": []}

db_embeddings = np.array(db["embeddings"]) if db["embeddings"] else np.zeros((0,128))
db_labels = np.array(db["labels"]) if db["labels"] else np.array([])

# ----------------------------
# Tracking buffers
# ----------------------------
next_face_id = 0
faces = {}  # face_id -> {'bbox','embedding','names','center'}

# ----------------------------
# Pygame GUI State
# ----------------------------
current_screen = "menu"
typing_name = ""
result_text = ""
camera_frame = None
stop_camera = False

# ----------------------------
# GUI Button
# ----------------------------
class Button:
    def __init__(self, rect, text, action):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.action = action
    def draw(self, surf):
        color = BUTTON_HOVER if self.rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(surf, color, self.rect)
        txt_surf = font.render(self.text, True, TEXT_COLOR)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surf.blit(txt_surf, txt_rect)
    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            self.action()

# ----------------------------
# Camera Worker
# ----------------------------
def camera_worker(train_mode=False, person_name=None):
    global camera_frame, result_text, stop_camera, db_embeddings, db_labels
    cap = cv2.VideoCapture(0)
    count = 0
    person_dir = os.path.join(DATA_DIR, person_name) if person_name else None
    if person_dir and not os.path.exists(person_dir):
        os.makedirs(person_dir)

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            continue
        camera_frame = cv2.flip(frame, 1)
        h,w = frame.shape[:2]

        # Detect faces
        blob = cv2.dnn.blobFromImage(camera_frame,1.0,(300,300),(104.0,177.0,123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf<0.5:
                continue
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)

            face_img = camera_frame[y1:y2,x1:x2]
            if face_img.size==0:
                continue

            face_resized = cv2.resize(face_img,(96,96))
            face_blob = cv2.dnn.blobFromImage(face_resized,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            embedder.setInput(face_blob)
            vec = embedder.forward().flatten()

            if train_mode:
                # Save face image and embedding
                count += 1
                img_path = os.path.join(person_dir,f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                db["embeddings"].append(vec)
                db["labels"].append(person_name)
                db_embeddings = np.array(db["embeddings"])
                db_labels = np.array(db["labels"])
                result_text = f"Treinando {person_name} | Captured: {count}"
            else:
                # Recognition: compare to DB
                if db_embeddings.shape[0]==0:
                    recognized_name = "Unknown"
                else:
                    dists = np.linalg.norm(db_embeddings - vec,axis=1)
                    min_idx = np.argmin(dists)
                    min_dist = dists[min_idx]
                    recognized_name = db_labels[min_idx] if min_dist<DIST_THRESHOLD else "Unknown"
                result_text = f"Reconhecido como: {recognized_name}" if recognized_name!="Unknown" else "Não reconhecido"

            # Draw rectangle and label
            color = (0,255,0) if train_mode or recognized_name!="Unknown" else (0,0,255)
            cv2.rectangle(camera_frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(camera_frame,f"{recognized_name if not train_mode else person_name}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        pygame.time.wait(10)

    cap.release()
    if train_mode:
        # Save DB after training
        with open(DB_FILE,"wb") as f:
            pickle.dump(db,f)
        result_text = f"Salvo como: {person_name}"
    stop_camera = False

# ----------------------------
# Button Actions
# ----------------------------
def start_train():
    global current_screen, typing_name, stop_camera
    typing_name=""
    current_screen="ask_name"
    stop_camera=False

def start_recognize():
    global current_screen, stop_camera
    current_screen="camera"
    stop_camera=False
    threading.Thread(target=camera_worker, args=(False,),daemon=True).start()

def go_back():
    global current_screen, stop_camera, result_text
    stop_camera=True
    current_screen="menu"
    result_text=""

# ----------------------------
# Buttons
# ----------------------------
menu_buttons=[
    Button((WIDTH//2-100,150,200,60),"Treinar",start_train),
    Button((WIDTH//2-100,300,200,60),"Reconhecer",start_recognize)
]
back_button = Button((WIDTH//2-100,HEIGHT-100,200,50),"Voltar",go_back)

# ----------------------------
# Main Loop
# ----------------------------
running=True
while running:
    screen.fill(BG_COLOR)
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            stop_camera=True
            pygame.quit()
            exit()
        elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            if current_screen=="menu":
                for b in menu_buttons:
                    b.check_click(event.pos)
            elif current_screen=="result":
                back_button.check_click(event.pos)
        elif current_screen=="ask_name" and event.type==pygame.KEYDOWN:
            if event.key==pygame.K_RETURN and typing_name.strip()!="":
                threading.Thread(target=camera_worker,args=(True,typing_name.strip()),daemon=True).start()
                current_screen="camera"
            elif event.key==pygame.K_BACKSPACE:
                typing_name=typing_name[:-1]
            else:
                typing_name+=event.unicode
        elif current_screen=="camera" and event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE:
            stop_camera=True
            current_screen="result"

    # ---------------- DRAW SCREENS ----------------
    if current_screen=="menu":
        title = font.render("Face Recognition App",True,TEXT_COLOR)
        screen.blit(title,(WIDTH//2 - title.get_width()//2,50))
        for b in menu_buttons:
            b.draw(screen)

    elif current_screen=="ask_name":
        txt = font.render(f"Digite o nome: {typing_name}_",True,TEXT_COLOR)
        screen.blit(txt,(WIDTH//2 - txt.get_width()//2, HEIGHT//2))

    elif current_screen=="camera":
        if camera_frame is not None:
            frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  # axes swap for Pygame
            surf = pygame.surfarray.make_surface(frame_rgb)
            surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
            screen.blit(surf, (0, 0))
        info_txt=font.render("Press ESC to finish",True,(255,255,0))
        screen.blit(info_txt,(10,10))
        if result_text:
            res_txt=font.render(result_text,True,(255,255,0))
            screen.blit(res_txt,(10,40))

    elif current_screen=="result":
        txt = font.render(result_text, True, TEXT_COLOR)
        screen.blit(txt,(WIDTH//2 - txt.get_width()//2, HEIGHT//2))
        back_button.draw(screen)

    pygame.display.flip()
    clock.tick(30)
