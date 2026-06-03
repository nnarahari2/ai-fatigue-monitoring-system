# AI Fatigue Monitoring System

## Overview

The AI Fatigue Monitoring System is a real-time computer vision and machine learning application designed to monitor driver alertness using a webcam. The system detects signs of drowsiness and driver distraction by analyzing facial landmarks, eye movements, and head orientation.

The project uses OpenCV and MediaPipe Face Mesh for facial landmark tracking and includes machine learning models trained on a custom driver-state dataset.

---

## Features

### Face Tracking

- Real-time facial landmark detection using MediaPipe Face Mesh
- Continuous face monitoring through a webcam feed

### Eye Landmark Tracking

- Tracks key eye landmarks in real time
- Calculates Eye Aspect Ratio (EAR) to estimate eye openness

### Drowsiness Detection

- Detects prolonged eye closure using EAR
- Displays visual alerts when drowsiness is detected
- Supports audio warning notifications

### Distraction Detection

- Estimates head orientation using facial landmarks
- Detects when the driver looks away from the road
- Generates distraction alerts in real time

### Event Logging

- Records fatigue and distraction events
- Stores event timestamps in CSV format
- Supports future analysis and model improvement

### Machine Learning Pipeline

- Custom dataset collection using webcam images
- Feature extraction using MediaPipe Face Mesh
- Training and evaluation of machine learning classifiers
- Model persistence using Joblib

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
│   ├── system_architecture.png
│   └── fatigue_log.csv
│
├── scripts/
│   ├── webcam_test.py
│   ├── fatigue_detection.py
│   ├── distraction_detection.py
│   ├── live_monitoring.py
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

A custom dataset was collected using webcam images during project development.

### Dataset Summary

| Class | Images |
|---------|---------:|
| Alert | 50 |
| Drowsy | 51 |
| Looking Left | 52 |
| Looking Right | 51 |
| **Total** | **204** |

---

## Machine Learning Experiments

Two machine learning approaches were evaluated.

### 1. Image-Based Classifier

**Features**

- 64 × 64 RGB images
- Pixel-based representation

**Model**

- Random Forest Classifier

**Results**

| Metric | Score |
|----------|----------:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

Saved Model:

```text
models/driver_state_classifier.pkl
```

---

### 2. Landmark-Based Classifier

**Features**

- Eye Aspect Ratio (EAR)
- Head Pose Offset
- Facial Landmark Geometry
- Nose Position
- Eye Landmark Positions
- Mouth Landmark Position

**Model**

- Random Forest Classifier

**Results**

| Metric | Score |
|----------|----------:|
| Accuracy | 93.3% |
| Precision | 94% |
| Recall | 93% |
| F1 Score | 93% |

Saved Model:

```text
models/landmark_driver_classifier.pkl
```

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

- Integrate the landmark-based classifier into the live monitoring pipeline
- Expand the dataset with additional participants
- Perform cross-subject evaluation
- Investigate deep learning approaches
- Develop a real-time driver analytics dashboard
- Deploy the system on embedded hardware

---

## Author

**Nitya Narahari**

Computer Science and Engineering  
University of California, Merced