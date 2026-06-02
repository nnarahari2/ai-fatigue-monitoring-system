import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_PATH = "dataset"

X = []
y = []

labels = [
    "alert",
    "drowsy",
    "looking_left",
    "looking_right"
]

for label in labels:

    folder = os.path.join(DATASET_PATH, label)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (64, 64))

        features = image.flatten()

        X.append(features)
        y.append(label)

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

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/driver_state_classifier.pkl"
)

print("\nModel saved to models/driver_state_classifier.pkl")