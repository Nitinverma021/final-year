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
