import cv2
import os
import numpy as np
import pickle
from imutils import paths
import streamlit as st
from datetime import datetime
import pandas as pd
import collections
from collections.abc import Iterable

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def process_embeddings():
    DATASET_DIR = "dataset"
    EMBEDDINGS_FILE = "output/embeddings.pickle"
    MODEL_FILE = "openface_nn4.small2.v1.t7"

    # Load the pre-trained model for embeddings
    embedder = cv2.dnn.readNetFromTorch(MODEL_FILE)

    # Initialize data storage
    known_embeddings = []
    known_names = []

    # Process each image in the dataset
    image_paths = list(paths.list_images(DATASET_DIR))

    if len(image_paths) == 0:
        raise ValueError("No images found in dataset. Please register faces first.")

    for (i, image_path) in enumerate(image_paths):
        # Extract the person's name from the image path
        name = image_path.split(os.path.sep)[-2]

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert the image to RGB and resize
        image = cv2.resize(image, (300, 300))
        (h, w) = image.shape[:2]

        # Construct a blob from the image
        image_blob = cv2.dnn.blobFromImage(
            image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
        )

        # Compute the 128-d embedding
        embedder.setInput(image_blob)
        vec = embedder.forward()

        # Append the embedding and name
        known_embeddings.append(vec.flatten())
        known_names.append(name)

    # Save the embeddings and names to a file
    data = {"embeddings": known_embeddings, "names": known_names}
    with open(EMBEDDINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))

    return len(known_embeddings), len(set(known_names))

def train_model():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC

    embeddingFile = "output/embeddings.pickle"
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"

    data = pickle.loads(open(embeddingFile, "rb").read())
    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    with open(recognizerFile, "wb") as f:
        f.write(pickle.dumps(recognizer))

    with open(labelEncFile, "wb") as f:
        f.write(pickle.dumps(labelEnc))

def mark_attendance():
    embeddingFile = "output/embeddings.pickle"
    embeddingModel = "openface_nn4.small2.v1.t7"
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"
    conf = 0.5

    # Load face detector
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Load face recognizer
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)
    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())

    # Initialize lists for attendance
    name_list = []
    rollno_list = []
    accuracy_list = []
    time_list = []

    # Start video capture
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)

    timeout = time.time() + 60
    count_num = 0

    while time.time() < timeout:
        ret, frame = cam.read()
        if not ret:
            st.error("Unable to capture frame. Please check your camera.")
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        
        # Create blob and detect faces
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        detector.setInput(imageBlob)
        detections = detector.forward()

        # Process each detected face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                # Get face embedding
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Predict
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # Get roll number from student.csv
                Roll_Number = ""
                with open('student.csv', 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    for row in reader:
                        if name in row:
                            Roll_Number = row[1]
                            break

                # Record attendance if confidence is high enough
                if (round(proba * 100)) > 70:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    name_list.append(name)
                    rollno_list.append(Roll_Number)
                    time_list.append(dtString)
                    accuracy_list.append(round(proba * 100))

                # Draw rectangle and text
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Display the frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

        # Update progress
        st.progress((timeout - time.time()) / 60)
        
        time.sleep(0.1)

    cam.release()

    # Process attendance data
    if name_list:
        counter = collections.Counter(rollno_list)
        pre_rollno = []
        for nam in counter:
            if (round((counter[nam]/len(name_list))*100)) > 10:
                pre_rollno.append(int(nam))

        # Create attendance records
        dict_mark = {'rollno': rollno_list, 'name': name_list, 'time': time_list, 'accuracy': accuracy_list}
        df = pd.DataFrame(dict_mark)
        df.to_csv('file2.csv', index=False)

        # Read student data
        student_data = pd.read_csv("student.csv")
        student_data = student_data.to_dict(orient="records")

        # Process present/absent lists
        namePresent = []
        roll_absent = []
        name_absent = []
        for dic in student_data:
            if dic["Roll_No"] in pre_rollno:
                namePresent.append(dic["name"])
            else:
                name_absent.append(dic["name"])
                roll_absent.append(dic["Roll_No"])

        # Create attendance DataFrame
        dict_present = {"Roll_no": pre_rollno, "name": namePresent, "attendance": ["P" for i in pre_rollno]}
        df2 = pd.DataFrame(dict_present)
        dict_absent = {"Roll_no": roll_absent, "name": name_absent, "attendance": ["A" for i in roll_absent]}
        df3 = pd.DataFrame(dict_absent)
        df4 = pd.concat([df2, df3], ignore_index=True)
        df4 = df4.sort_values(by=['Roll_no'])

        # Save attendance record
        now2 = datetime.now()
        date_str = now2.strftime("%d-%m-%Y")
        df4.to_csv(f"{date_str}.csv", index=False)

        return df4
    else:
        st.warning("No faces were recognized during the session.")
        return None 