# AI Fatigue Monitoring System

## Overview

The AI Fatigue Monitoring System is a real-time computer vision application designed to monitor driver alertness using a webcam. The system detects signs of drowsiness and driver distraction by analyzing facial landmarks, eye movements, and head orientation.

The project uses OpenCV and MediaPipe Face Mesh to perform real-time facial landmark tracking and generates alerts when fatigue or distraction is detected.

This project was developed to explore the use of computer vision and machine learning techniques for real-time driver safety monitoring and human attention analysis.

---

## Features

### Face Tracking

* Real-time facial landmark detection using MediaPipe Face Mesh
* Continuous face monitoring through a webcam feed

### Eye Landmark Tracking

* Tracks key eye landmarks in real time
* Calculates Eye Aspect Ratio (EAR) to estimate eye openness

### Drowsiness Detection

* Detects prolonged eye closure using EAR
* Displays visual alerts when drowsiness is detected
* Supports audio warning notifications

### Distraction Detection

* Estimates head orientation using facial landmarks
* Detects when the driver looks away from the road
* Generates distraction alerts in real time

### Live Monitoring

* Combines fatigue and distraction detection into a single monitoring system
* Provides continuous driver status feedback

---

## Technologies Used

* Python
* OpenCV
* MediaPipe Face Mesh
* NumPy
* SciPy

---

## Project Structure

```text
AI-FATIGUE-MONITORING-SYSTEM
│
├── demo/
│   ├── face_tracking_demo.png
│   ├── eye_landmark_tracking.png
│   ├── drowsiness_detection_demo.png
│   └── distraction_detection.png
│
├── scripts/
│   ├── webcam_test.py
│   ├── fatigue_detection.py
│   ├── distraction_detection.py
│   └── live_monitoring.py
│
├── results/
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

### Live Monitoring System

```bash
python scripts/live_monitoring.py
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

* Event logging for fatigue and distraction incidents
* Dataset collection and analysis
* Machine learning classification models
* Performance evaluation using accuracy, precision, and recall
* Enhanced driver monitoring dashboard

---

## Author

Nitya Narahari

Computer Science and Engineering, University of California, Merced