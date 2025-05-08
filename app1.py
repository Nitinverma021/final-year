import os
import cv2
import time
import imutils
import streamlit as st
import numpy as np
from datetime import datetime
from face_recognition import process_embeddings, train_model, mark_attendance
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Constants
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
DATASET_DIR = 'dataset'
MAX_IMAGES = 50
RESIZE_WIDTH = 400

# Load the face detector
detector = cv2.CascadeClassifier(CASCADE_FILE)

# Streamlit UI
st.title("Face Recognition Attendance System")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Register Face", "Mark Attendance"])

# --- Webcam Registration ---
class RegistrationTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_images = []
        self.face_count = 0
        self.name = None
        self.role_number = None
        self.output_path = None
        self.max_images = MAX_IMAGES
        self.detector = detector

    def set_user(self, name, role_number):
        self.name = name
        self.role_number = role_number
        self.output_path = os.path.join(DATASET_DIR, name)
        os.makedirs(self.output_path, exist_ok=True)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if self.face_count < self.max_images:
                face_img = img[y:y+h, x:x+w]
                face_filename = os.path.join(self.output_path, f"{str(self.face_count).zfill(5)}.png")
                cv2.imwrite(face_filename, face_img)
                self.face_count += 1
        return img

with tab1:
    st.header("Register New Face (Webcam)")
    name = st.text_input("Enter your Name:")
    role_number = st.text_input("Enter your Roll Number:")
    reg_transformer = RegistrationTransformer()
    if name and role_number:
        reg_transformer.set_user(name, role_number)
        st.info(f"Capturing up to {MAX_IMAGES} face images for {name}.")
        webrtc_ctx = webrtc_streamer(key="register", video_transformer_factory=lambda: reg_transformer)
        if reg_transformer.face_count >= MAX_IMAGES:
            st.success(f"Registration complete! Captured {reg_transformer.face_count} images for {name}.")
            st.write("Processing images...")
            try:
                num_embeddings, num_people = process_embeddings()
                train_model()
                st.success(f"Image processing complete! Processed {num_embeddings} images for {num_people} people.")
            except Exception as e:
                st.error(f"Error processing images: {str(e)}")

# --- Webcam Attendance ---
class AttendanceTransformer(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        self.last_attendance_df = None
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frames.append(img)
        self.frame_count += 1
        # Optionally, you can run recognition every N frames
        return img

with tab2:
    st.header("Mark Attendance (Webcam)")
    att_transformer = AttendanceTransformer()
    st.info("Click Start and show your face to the webcam.")
    webrtc_ctx2 = webrtc_streamer(key="attendance", video_transformer_factory=lambda: att_transformer)
    st.write(f"Frames captured: {att_transformer.frame_count}")
    if st.button("Process Attendance"):
        st.write("Processing captured frames for attendance...")
        if att_transformer.frames:
            last_frame = att_transformer.frames[-1]
            st.image(last_frame, channels="BGR")
            st.warning("You need to adapt mark_attendance to process this frame.")
        else:
            st.error("No frames captured. Please try again.")

# Remove any cv2.destroyAllWindows() calls as they're not needed in Streamlit
