from face.detector import FaceDetector
from face.embedder import FaceEmbedder
from face.database import FaceDatabase
from face.recognizer import Recognizer
from face.trainer import Trainer
from ui.app import App

def main():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = FaceDatabase()
    recognizer = Recognizer(db)
    trainer = Trainer(db)

    app = App(detector, embedder, db, recognizer, trainer)
    app.run()

if __name__ == "__main__":
    main()
