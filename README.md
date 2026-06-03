# AI Fatigue Monitoring System

## Overview

The AI Fatigue Monitoring System is a real-time computer vision and machine learning application designed to monitor driver alertness using a webcam. The system detects signs of drowsiness and driver distraction by analyzing facial landmarks, eye movements, and head orientation.

The project combines OpenCV, MediaPipe Face Mesh, and machine learning techniques to create a complete end-to-end driver monitoring pipeline, including data collection, feature engineering, model training, evaluation, and deployment.

---

## Key Results

- Real-time facial landmark tracking using MediaPipe Face Mesh
- Drowsiness detection using Eye Aspect Ratio (EAR)
- Driver distraction detection using head orientation analysis
- Custom dataset containing 204 labeled driver-state images
- Landmark-based machine learning classifier
- Random Forest model achieving **93.3% classification accuracy**
- Real-time machine learning inference integrated into the live monitoring system
- Event logging for fatigue and distraction incidents

---

## System Architecture

![System Architecture](results/system_architecture.png)

---

## Features

### Face Tracking

- Real-time facial landmark detection using MediaPipe Face Mesh
- Continuous face monitoring through a webcam feed

### Eye Landmark Tracking

- Tracks key eye landmarks in real time
- Calculates Eye Aspect Ratio (EAR) to estimate eye openness

### Drowsiness Detection

- Detects prolonged eye closure
- Generates visual fatigue alerts
- Supports audio warning notifications

### Distraction Detection

- Estimates head orientation using facial landmarks
- Detects when the driver looks away from the road
- Generates distraction alerts in real time

### Event Logging

- Records fatigue and distraction events
- Stores event timestamps in CSV format
- Enables future data analysis and evaluation

### Machine Learning Classification

- Custom dataset collection pipeline
- Landmark-based feature extraction
- Random Forest classifier training and evaluation
- Real-time prediction using a trained machine learning model

---

## Technologies Used

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- SciPy
- Scikit-Learn
- Joblib

---

## Project Structure

```text
AI-FATIGUE-MONITORING-SYSTEM
│
├── dataset/
│   ├── alert/
│   ├── drowsy/
│   ├── looking_left/
│   └── looking_right/
│
├── demo/
│   ├── face_tracking_demo.png
│   ├── eye_landmark_tracking.png
│   ├── drowsiness_detection_demo.png
│   └── distraction_detection.png
│
├── models/
│   ├── driver_state_classifier.pkl
│   └── landmark_driver_classifier.pkl
│
├── results/
│   └── system_architecture.png
│
├── scripts/
│   ├── webcam_test.py
│   ├── fatigue_detection.py
│   ├── distraction_detection.py
│   ├── live_monitoring.py
│   ├── live_monitoring_v1.py
│   ├── data_collection.py
│   ├── train_classifier.py
│   └── train_landmark_classifier.py
│
├── alert.wav
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ai-fatigue-monitoring-system
```

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Webcam Test

```bash
python scripts/webcam_test.py
```

### Drowsiness Detection

```bash
python scripts/fatigue_detection.py
```

### Distraction Detection

```bash
python scripts/distraction_detection.py
```

### Unified Driver Monitoring System

```bash
python scripts/live_monitoring.py
```

### Data Collection

```bash
python scripts/data_collection.py
```

### Train Image-Based Classifier

```bash
python scripts/train_classifier.py
```

### Train Landmark-Based Classifier

```bash
python scripts/train_landmark_classifier.py
```

---

## Dataset

A custom driver-state dataset was collected using webcam images during project development.

### Dataset Summary

| Class | Images |
|---------|---------:|
| Alert | 50 |
| Drowsy | 51 |
| Looking Left | 52 |
| Looking Right | 51 |
| **Total** | **204** |

---

## Machine Learning Pipeline

### Dataset Collection

A custom dataset was created by capturing webcam images across four driver states:

- Alert
- Drowsy
- Looking Left
- Looking Right

### Feature Engineering

The landmark-based classifier uses facial landmark features extracted from MediaPipe Face Mesh:

- Eye Aspect Ratio (EAR)
- Head Direction Offset
- Nose Coordinates
- Left Eye Coordinates
- Right Eye Coordinates
- Mouth Coordinates

### Model Training

Algorithm:

- Random Forest Classifier

### Model Evaluation

The landmark-based model was evaluated on a held-out test set.

| Metric | Score |
|----------|----------:|
| Accuracy | 93.3% |
| Precision | 94% |
| Recall | 93% |
| F1 Score | 93% |

### Trained Models

```text
models/driver_state_classifier.pkl
models/landmark_driver_classifier.pkl
```

---

## Technical Highlights

- Real-time computer vision pipeline
- Facial landmark extraction using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) computation
- Head pose estimation
- Driver distraction detection
- Fatigue detection
- Dataset collection and labeling
- Machine learning model training
- Model evaluation and deployment
- Real-time ML inference
- Event logging and monitoring

---

## Demo Results

### Face Tracking

![Face Tracking](demo/face_tracking_demo.png)

### Eye Landmark Tracking

![Eye Landmark Tracking](demo/eye_landmark_tracking.png)

### Drowsiness Detection

![Drowsiness Detection](demo/drowsiness_detection_demo.png)

### Distraction Detection

![Distraction Detection](demo/distraction_detection.png)

---

## Future Improvements

- Expand the dataset with multiple participants
- Improve left/right classification robustness
- Explore deep learning architectures
- Deploy on embedded hardware platforms
- Build a driver analytics dashboard
- Evaluate performance under varying lighting conditions
- Integrate cloud-based event logging

---

## Summary

This project demonstrates the complete machine learning lifecycle:

1. Data Collection
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Real-Time Deployment

The system combines computer vision and machine learning techniques to perform real-time driver monitoring, fatigue detection, and distraction analysis using a webcam.

---

## Author

**Nitya Narahari**

Computer Science and Engineering  
University of California, Merced