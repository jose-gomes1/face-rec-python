# ui/app.py

import pygame
import cv2
import numpy as np
import asyncio

import config
from ui.button import Button
from ui.camera_worker import CameraWorker

class App:
    def __init__(self, detector, embedder, db, recognizer, trainer):
        pygame.init()
        self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        pygame.display.set_caption("Face Recognition")

        self.detector = detector
        self.embedder = embedder
        self.db = db
        self.recognizer = recognizer
        self.trainer = trainer

        self.font = pygame.font.SysFont(None, config.FONT_SIZE)
        self.camera_frame = None
        self.worker = None
        self.state = "menu"
        self.name_input = ""

        self.buttons = [
            Button((config.WIDTH//2 -100, 150, 200,60), "Train", self.to_train),
            Button((config.WIDTH//2 -100, 300, 200,60), "Recognize", self.to_recognize)
        ]

    def to_train(self):
        self.state = "type_name"

    def to_recognize(self):
        self.start_camera(train=False)
    
    # Inside App class
    def update_frame(self, frame, text):
        self.camera_frame = frame
        self.result_text = text


    def start_camera(self, train=False, name=None):
        self.state = "camera"
        self.worker = CameraWorker(
            self.detector,
            self.embedder,
            self.db,
            self.recognizer,
            self.trainer,
            callback=self.update_frame,
            train_name=name
        )
        self.worker.start()

    def update_frame(self, frame, result_text):
        self.camera_frame = frame
        self.result_text = result_text

    def stop_camera(self):
        if self.worker:
            self.worker.stop_flag = True
            self.worker.join()
            self.worker = None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_camera()
                    asyncio.sleep(0.1)
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.state == "menu":
                        for b in self.buttons: b.click(event.pos)

                if self.state == "type_name":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN and self.name_input:
                            self.start_camera(train=True, name=self.name_input)
                        elif event.key == pygame.K_BACKSPACE:
                            self.name_input = self.name_input[:-1]
                        else:
                            self.name_input += event.unicode

                if self.state == "camera":
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.stop_camera()
                        self.state = "menu"

            self.draw()
            pygame.display.update()

    def draw(self):
        self.screen.fill(config.BG_COLOR)

        if self.state == "menu":
            title = self.font.render("Face Recognition App", True, config.TEXT_COLOR)
            self.screen.blit(title, (config.WIDTH//2 - title.get_width()//2, 50))
            for b in self.buttons:
                b.draw(self.screen)

        elif self.state == "type_name":
            text = self.font.render(f"Enter name: {self.name_input}", True, config.TEXT_COLOR)
            self.screen.blit(text, (config.WIDTH//2 - text.get_width()//2, config.HEIGHT//2))

        elif self.state == "camera" and self.camera_frame is not None:
            if self.camera_frame is not None:
                frame_rgb = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
                frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  # axes swap for Pygame
                surf = pygame.surfarray.make_surface(frame_rgb)
                surf = pygame.transform.scale(surf, (config.WIDTH, config.HEIGHT))
                self.screen.blit(surf, (0, 0))
            info_txt=self.font.render("Press ESC to finish",True,(255,255,0))
            self.screen.blit(info_txt,(10,10))
            if self.result_text:
                text_surf = self.font.render(self.result_text, True, (255,255,0))
                self.screen.blit(text_surf, (10, 10))