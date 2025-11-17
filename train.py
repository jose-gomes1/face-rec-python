import cv2
import os
import time

def create_or_update_person_dataset(base_path="dataset"):
    # Ask user for name
    person_name = input("Enter the name of the person: ").strip()

    # Create base dataset directory if not exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Person folder
    person_dir = os.path.join(base_path, person_name)

    # Create or update?
    if os.path.exists(person_dir):
        print(f"[INFO] Updating existing dataset for '{person_name}'.")
    else:
        print(f"[INFO] Creating new dataset for '{person_name}'.")
        os.makedirs(person_dir)

    return person_dir


def capture_images(person_dir, required_count=500):
    cam = cv2.VideoCapture(0)

    # Check camera
    if not cam.isOpened():
        print("Error: Camera not found.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0

    print("\n[INFO] Starting image capture... Look at the camera.")
    time.sleep(1)

    while count < required_count:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles & save faces
        for (x, y, w, h) in faces:
            count += 1

            # Crop and save
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)

            # Display progress
            cv2.putText(frame, f"Images: {count}/{required_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Training Camera - Press Q to Quit", frame)

        # Quit early if needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\n[INFO] Captured {count} images.")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_dir = create_or_update_person_dataset()
    capture_images(person_dir)
