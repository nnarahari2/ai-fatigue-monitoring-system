# AI Fatigue + Distraction Monitoring System

## Overview

The AI Fatigue + Distraction Monitoring System is a real-time computer vision platform designed to monitor driver attentiveness and detect potentially unsafe behavior through live facial perception and behavioral analysis.

The system performs real-time webcam inference using MediaPipe Face Mesh, OpenCV, and geometric facial landmark analysis to identify signs of drowsiness and distraction. By combining Eye Aspect Ratio (EAR) fatigue estimation with head pose-based distraction monitoring, the platform generates live safety alerts capable of identifying prolonged eye closure and directional attention drift.

This project demonstrates the integration of real-time perception pipelines, human-centered AI monitoring, and intelligent safety analysis within a unified computer vision system.

---

# Features

- Real-time webcam inference pipeline
- Facial landmark extraction using MediaPipe Face Mesh
- Eye landmark tracking and geometric analysis
- Eye Aspect Ratio (EAR)-based fatigue detection
- Head pose estimation for distraction monitoring
- Real-time driver attention analysis
- Unified multimodal safety monitoring system
- Live visual alert generation
- Real-time facial mesh visualization
- Modular computer vision architecture

---

# System Architecture

![System Architecture](results/system_architecture.png)

---

# Demo Results

## Facial Landmark Tracking

![Face Tracking](demo/face_tracking_demo.png)

---

## Eye Landmark Tracking

![Eye Tracking](demo/eye_landmark_tracking.png)

---

## Drowsiness Detection

![Drowsiness Detection](demo/drowsiness_detection_demo.png)

---

## Distraction Detection

![Distraction Detection](demo/distraction_detection.png)

---

# System Pipeline

```text
Webcam Input
    ↓
Face Detection
    ↓
MediaPipe Facial Landmark Tracking
    ↓
Eye Landmark Extraction + Head Pose Analysis
    ↓
EAR Fatigue Analysis + Distraction Monitoring
    ↓
Real-Time Safety Alert Generation
```

---

# Technologies Used

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- SciPy

---

# Fatigue Detection Methodology

The fatigue monitoring subsystem utilizes Eye Aspect Ratio (EAR) analysis to estimate eye openness over consecutive frames.

EAR is computed using geometric distances between key eye landmarks extracted from MediaPipe Face Mesh. When the EAR value remains below a predefined threshold for multiple consecutive frames, the system classifies the behavior as potential drowsiness and generates a real-time fatigue alert.

This approach enables lightweight, real-time fatigue estimation without requiring deep neural network inference.

---

# Distraction Detection Methodology

The distraction monitoring subsystem performs lightweight head pose estimation using geometric facial landmark relationships.

The system estimates directional attention drift by comparing the nose landmark position relative to the horizontal facial centerline. Significant directional offsets are interpreted as attention deviation and classified as distraction events.

The system is capable of detecting:

- Looking left
- Looking right
- Attention drift
- Forward attentive state

---

# Installation

```bash
pip install opencv-python mediapipe numpy scipy
```

---

# Running the System

## Unified Monitoring System

```bash
python scripts/live_monitoring.py
```

---

## Fatigue Detection Module

```bash
python scripts/fatigue_detection.py
```

---

## Distraction Detection Module

```bash
python scripts/distraction_detection.py
```

---

# Repository Structure

```text
AI-FATIGUE-MONITORING-SYSTEM/
│
├── demo/
│   ├── distraction_detection.png
│   ├── drowsiness_detection_demo.png
│   ├── eye_landmark_tracking.png
│   └── face_tracking_demo.png
│
├── results/
│   └── system_architecture.png
│
├── scripts/
│   ├── distraction_detection.py
│   ├── fatigue_detection.py
│   ├── live_monitoring.py
│   └── webcam_test.py
│
├── README.md
└── .gitignore
```

---

# Applications

This project aligns with several intelligent systems and real-time AI application domains, including:

- Driver monitoring systems
- Autonomous vehicle safety systems
- Human attention analysis
- Intelligent transportation systems
- Robotics perception pipelines
- Real-time human behavior monitoring
- Computer vision safety applications

---

# Future Improvements

- Audio-based warning systems
- Mobile phone distraction detection
- Deep learning-based gaze estimation
- Fatigue analytics dashboard
- Multi-person monitoring support
- Temporal attention scoring
- Driver behavior analytics
- Edge-device deployment optimization

---

# Author

Nitya Narahari

Computer Science & Engineering — UC Merced

Focused on:

- Computer Vision
- Intelligent Systems
- Real-Time AI
- Robotics Perception
- Safety-Oriented AI Systems