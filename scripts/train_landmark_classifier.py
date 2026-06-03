import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------
# MediaPipe Setup
# ---------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# ---------------------------------------------------
# Landmark Indices
# ---------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454

LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 263
MOUTH_CENTER = 13

# ---------------------------------------------------
# EAR Calculation
# ---------------------------------------------------
def eye_aspect_ratio(points):

    vertical_1 = distance.euclidean(points[1], points[5])
    vertical_2 = distance.euclidean(points[2], points[4])

    horizontal = distance.euclidean(points[0], points[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
DATASET_PATH = "dataset"

labels = [
    "alert",
    "drowsy",
    "looking_left",
    "looking_right"
]

X = []
y = []

# ---------------------------------------------------
# Process Images
# ---------------------------------------------------
for label in labels:

    folder = os.path.join(DATASET_PATH, label)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0]

        h, w, _ = image.shape

        left_eye = []
        right_eye = []

        for idx in LEFT_EYE:

            x = landmarks.landmark[idx].x * w
            y_coord = landmarks.landmark[idx].y * h

            left_eye.append((x, y_coord))

        for idx in RIGHT_EYE:

            x = landmarks.landmark[idx].x * w
            y_coord = landmarks.landmark[idx].y * h

            right_eye.append((x, y_coord))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2

        nose_x = landmarks.landmark[NOSE_TIP].x
        nose_y = landmarks.landmark[NOSE_TIP].y

        left_face_x = landmarks.landmark[LEFT_FACE].x
        right_face_x = landmarks.landmark[RIGHT_FACE].x

        face_center = (left_face_x + right_face_x) / 2

        head_offset = nose_x - face_center

        left_eye_x = landmarks.landmark[LEFT_EYE_CENTER].x
        left_eye_y = landmarks.landmark[LEFT_EYE_CENTER].y

        right_eye_x = landmarks.landmark[RIGHT_EYE_CENTER].x
        right_eye_y = landmarks.landmark[RIGHT_EYE_CENTER].y

        mouth_x = landmarks.landmark[MOUTH_CENTER].x
        mouth_y = landmarks.landmark[MOUTH_CENTER].y

        features = [
            avg_ear,
            head_offset,
            nose_x,
            nose_y,
            left_eye_x,
            left_eye_y,
            right_eye_x,
            right_eye_y,
            mouth_x,
            mouth_y
        ]

        X.append(features)
        y.append(label)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(
    model,
    "models/landmark_driver_classifier.pkl"
)

print("\nSaved model:")
print("models/landmark_driver_classifier.pkl")