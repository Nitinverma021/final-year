import os
import cv2
import time
import imutils

# Constants
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
DATASET_DIR = 'dataset'
MAX_IMAGES = 50
RESIZE_WIDTH = 400

# Load the face detector
detector = cv2.CascadeClassifier(CASCADE_FILE)

# Input user details
name = input("Enter your Name: ").strip()
role_number = input("Enter your Roll Number: ").strip()

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
    
import cv2
import os
import numpy as np
import pickle
from imutils import paths

# Paths
DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "output/embeddings.pickle"
MODEL_FILE = "openface_nn4.small2.v1.t7"

# Load the pre-trained model for embeddings
print("[INFO] Loading pre-trained face embedding model...")
embedder = cv2.dnn.readNetFromTorch(MODEL_FILE)

# Initialize data storage
known_embeddings = []
known_names = []

# Process each image in the dataset
print("[INFO] Quantifying faces...")
image_paths = list(paths.list_images(DATASET_DIR))

if len(image_paths) == 0:
    raise ValueError("[ERROR] No images found in dataset. Ensure dataset directory is populated.")

for (i, image_path) in enumerate(image_paths):
    print(f"[INFO] Processing image {i + 1}/{len(image_paths)}: {image_path}")

    # Extract the person's name from the image path
    name = image_path.split(os.path.sep)[-2]

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Unable to read image: {image_path}. Skipping.")
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

print(f"[INFO] Collected {len(known_embeddings)} embeddings for {len(set(known_names))} unique individuals.")

# Save the embeddings and names to a file
print("[INFO] Serializing embeddings...")
data = {"embeddings": known_embeddings, "names": known_names}
with open(EMBEDDINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Embeddings saved to:", EMBEDDINGS_FILE)

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


def training():

    #initilizing of embedding & recognizer
    embeddingFile = "output/embeddings.pickle"
    #New & Empty at initial
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"

    print("Loading face embeddings...")
    data = pickle.loads(open(embeddingFile, "rb").read())

    print("Encoding labels...")
    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["names"])


    print("Training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open(recognizerFile, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open(labelEncFile, "wb")
    f.write(pickle.dumps(labelEnc))
    f.close()

training()

from collections.abc import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from datetime import datetime
import pandas as pd
import collections

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

timeout = time.time()+ 60
count_num=0
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

print("[INFO] loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

Roll_Number = ""
box = []
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
name_list = []
rollno_list = []
accuracy_list =[]
time_list = []
while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        count_num += 1
                        print(name)
                        print(proba*100)

                listString = str(box)
                if name in listString:
                    singleList = list(flatten(box))
                    listlen = len(singleList)
                    Index = singleList.index(name)
                    name = singleList[Index]
                    Roll_Number = singleList[Index + 1]
                    print(Roll_Number)
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    if (round(proba*100))>70:
                        name_list.append(name)
                        rollno_list.append(Roll_Number)
                        time_list.append(dtString)
                        accuracy_list.append((round(proba*100)))

            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or time.time()>timeout:
        break
    time.sleep(0.3)
dict_mark = {'rollno': rollno_list, 'name': name_list, 'time': time_list, 'accuracy': accuracy_list}
counter = collections.Counter(rollno_list)
pre_rollno = []
for nam in counter:
    print(f"{nam}  ->  {round((counter[nam]/len(name_list))*100)}")
    if (round((counter[nam]/len(name_list))*100))> 10:
        pre_rollno.append(int(nam))

df = pd.DataFrame(dict_mark)
df.to_csv('file2.csv', index=False)
student_data = pd.read_csv("student.csv")
student_data = student_data.to_dict(orient="records")

now2 = datetime.now()
date_str = now2.strftime("%d-%m-%Y")
student_data = pd.read_csv("student.csv")
student_data = student_data.to_dict(orient="records")

namePresent = []
roll_absent = []
name_absent = []
for dic in student_data:
    if dic["Roll_No"] in pre_rollno:
        namePresent.append(dic["name"])
    else:
        name_absent.append(dic["name"])
        roll_absent.append(dic["Roll_No"])
dict_present = {"Roll_no": pre_rollno, "name": namePresent, "attendance": ["P" for i in pre_rollno]}
df2 = pd.DataFrame(dict_present)
dict_absent = {"Roll_no": roll_absent, "name": name_absent, "attendance": "A"}
df3 = pd.DataFrame(dict_absent)
df4 = pd.concat([df2, df3], ignore_index=True)
df4.reset_index()
df4 = df4.sort_values(by=['Roll_no'])
df4.to_csv(f"{date_str}.csv", index=False)
#print(count_num)
print(df4)
cam.release()
cv2.destroyAllWindows()