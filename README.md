# AI Fatigue Monitoring System

## Overview

The AI Fatigue Monitoring System is a real-time computer vision and machine learning application designed to monitor driver alertness using a webcam. The system detects signs of drowsiness and driver distraction by analyzing facial landmarks, eye movements, and head orientation.

The project uses OpenCV and MediaPipe Face Mesh for real-time facial landmark tracking and includes a machine learning pipeline for driver state classification.

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
- Stores event timestamps in a CSV log file
- Enables future data analysis and evaluation

### Machine Learning Classification

- Custom dataset collected using webcam images
- Random Forest classifier trained on driver state images
- Supports classification of:
  - Alert
  - Drowsy
  - Looking Left
  - Looking Right

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
│   └── driver_state_classifier.pkl
│
├── results/
│   ├── fatigue_log.csv
│   └── system_architecture.png
│
├── scripts/
│   ├── webcam_test.py
│   ├── fatigue_detection.py
│   ├── distraction_detection.py
│   ├── live_monitoring.py
│   ├── data_collection.py
│   └── train_classifier.py
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

### Train Machine Learning Classifier

```bash
python scripts/train_classifier.py
```

---

## Dataset

A custom driver-state dataset was collected using webcam images.

Dataset Summary:

| Class | Images |
|---------|---------|
| Alert | 50 |
| Drowsy | 51 |
| Looking Left | 52 |
| Looking Right | 51 |
| **Total** | **204** |

---

## Machine Learning Results

A Random Forest classifier was trained on the collected dataset.

### Evaluation Metrics

| Metric | Score |
|----------|----------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

### Model Output

The trained model is stored in:

```text
models/driver_state_classifier.pkl
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

- Larger and more diverse dataset collection
- Multiple-user testing
- Deep learning-based classification models
- Real-world driving dataset evaluation
- Dashboard for driver analytics
- Cloud-based event logging and monitoring

---

## Author

**Nitya Narahari**

Computer Science and Engineering  
University of California, Merced