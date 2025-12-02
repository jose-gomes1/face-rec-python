# Face Recognition in Python

A simple Python project for detecting and recognizing faces using OpenCV and NumPy. This repository provides tools for building a face dataset, encoding faces, and recognizing them with a basic UI.

---

## 📂 Repository Structure

```
/dataset/       — Folder for storing face images for training/recognition  
/face/          — Face-processing code (detection, encoding, matching)  
/ui/            — User interface code  
config.py       — Configuration (paths, parameters, thresholds)  
main.py         — Main script to launch face recognition / UI  
.gitignore      — Files/folders ignored by Git  
README.md       — This file  
```

---

## ⚙️ Requirements

* Python 3.x
* `opencv-python`
* `numpy`
* `scipy`
* `pygame`

## Library instalation

```
pip install opencv-python
pip install scipy
pip install pygame 
```

---

## 🚀 Usage

1. Prepare a `dataset/` folder containing subfolders for each person. Each subfolder should include one or more images of that person.
2. Adjust parameters in `config.py` (e.g., paths, thresholds).
3. Run the main script:

```bash
python3 main.py
```

4. Recognized faces will be displayed via the UI and compared against the dataset.

---

## 🎯 Features

* Detects faces in images using OpenCV.
* Encodes faces for recognition.
* Compares detected faces against a known dataset.
* Displays results with a basic UI using Pygame.

---

## ⚠️ Limitations

* Dataset must be organized as one subfolder per identity.
* Recognition depends on image quality, lighting, and angle.
* Not optimized for multiple faces or large datasets.
* Designed for learning and experimentation, not production-level applications.

---

## 🧪 Example Dataset Structure

```
dataset/
  ├── mike/
  │     ├── 1.jpg
  │     └── 2.jpg
  └── mateus/
        └── 1.jpg
```

---

## 🔄 Comparison

This project provides a lightweight, customizable approach for learning purposes. For production-ready solutions, consider libraries like [face_recognition](https://github.com/ageitgey/face_recognition).

---

## 📄 License & Credits

This project is open-source and free to use. Credit to the open-source community for inspiration and foundational work in face detection and recognition.

---
