import os
import cv2
import time
import imutils
import streamlit as st

# Constants
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
DATASET_DIR = 'dataset'
MAX_IMAGES = 50
RESIZE_WIDTH = 400

# Load the face detector
detector = cv2.CascadeClassifier(CASCADE_FILE)

# Input user details
name = st.text_input("Enter your Name:")
role_number = st.text_input("Enter your Roll Number:")

# Create a folder for the dataset if it doesn't exist
output_path = os.path.join(DATASET_DIR, name)
os.makedirs(output_path, exist_ok=True)

print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

image_count = 0

try:
    while image_count < MAX_IMAGES:
        # Capture frame from webcam
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Unable to capture frame. Exiting...")
            break

        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=RESIZE_WIDTH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract the face region
            face = frame[y:y+h, x:x+w]

            # Save the face image to the dataset folder
            face_filename = os.path.join(output_path, f"{str(image_count).zfill(5)}.png")
            cv2.imwrite(face_filename, face)
            print(f"[INFO] Saved: {face_filename}")
            image_count += 1

        # Show the video stream
        cv2.imshow("Frame", frame)

        # Stop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Exiting...")
            break

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    print(f"[INFO] Collected {image_count} images for {name}.")
    cam.release()
    cv2.destroyAllWindows()
