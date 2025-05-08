import os
import cv2
import time
import imutils
import streamlit as st
import numpy as np
from datetime import datetime
from face_recognition import process_embeddings, train_model, mark_attendance

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

with tab1:
    st.header("Register New Face")
    name = st.text_input("Enter your Name:")
    role_number = st.text_input("Enter your Roll Number:")
    
    if st.button("Start Registration"):
        if name and role_number:
            # Create a folder for the dataset if it doesn't exist
            output_path = os.path.join(DATASET_DIR, name)
            os.makedirs(output_path, exist_ok=True)
            
            st.write("Starting camera...")
            cam = cv2.VideoCapture(0)
            time.sleep(2.0)
            
            image_count = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                while image_count < MAX_IMAGES:
                    ret, frame = cam.read()
                    if not ret:
                        st.error("Unable to capture frame. Please check your camera.")
                        break
                    
                    # Resize the frame for faster processing
                    frame = imutils.resize(frame, width=RESIZE_WIDTH)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    # Draw rectangle around faces and save images
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        face = frame[y:y+h, x:x+w]
                        face_filename = os.path.join(output_path, f"{str(image_count).zfill(5)}.png")
                        cv2.imwrite(face_filename, face)
                        image_count += 1
                        progress_bar.progress(image_count / MAX_IMAGES)
                        status_text.text(f"Captured {image_count} images")
                    
                    # Display the frame in Streamlit
                    st.image(frame, channels="BGR", use_column_width=True)
                    
                    if image_count >= MAX_IMAGES:
                        break
                    
                    time.sleep(0.1)  # Small delay to prevent overwhelming the system
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                cam.release()
                st.success(f"Registration complete! Captured {image_count} images for {name}.")
                
                # Process the captured images
                st.write("Processing images...")
                try:
                    # Process embeddings and train model
                    num_embeddings, num_people = process_embeddings()
                    train_model()
                    st.success(f"Image processing complete! Processed {num_embeddings} images for {num_people} people.")
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")

with tab2:
    st.header("Mark Attendance")
    if st.button("Start Attendance"):
        st.write("Starting face recognition...")
        try:
            attendance_df = mark_attendance()
            if attendance_df is not None:
                st.success("Attendance marked successfully!")
                st.dataframe(attendance_df)
        except Exception as e:
            st.error(f"Error marking attendance: {str(e)}")

# Remove any cv2.destroyAllWindows() calls as they're not needed in Streamlit
