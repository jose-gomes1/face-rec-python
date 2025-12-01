import cv2
import config

class FaceEmbedder:
    def __init__(self):
        # loads the .t7 pre-trained model
        self.model = cv2.dnn.readNetFromTorch(config.FACENET_T7)

    def get_embedding(self, face_img):
        # prepares the image
        resized = cv2.resize(face_img, (96,96))
        # blob creation: normalization, size, mean substraction (not used), convert BGR to RGB
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, (96,96), (0,0,0), swapRB=True)
        self.model.setInput(blob)
        # generates the embedding
        return self.model.forward().flatten()
