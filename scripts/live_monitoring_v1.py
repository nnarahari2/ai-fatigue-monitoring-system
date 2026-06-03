import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import threading
import os
import csv
from datetime import datetime

# ---------------------------------------------------
# Audio Alert Function
# ---------------------------------------------------
alarm_playing = False

def play_alarm():

    global alarm_playing

    if not alarm_playing:

        alarm_playing = True

        os.system('afplay alert.wav')

        alarm_playing = False

# ---------------------------------------------------
# Event Logging Function
# ---------------------------------------------------
def log_event(event):

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event
        ])

# ---------------------------------------------------
# Eye Aspect Ratio (EAR) Calculation
# ---------------------------------------------------
def eye_aspect_ratio(eye_points):

    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])

    horizontal = distance.euclidean(eye_points[0], eye_points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear

# ---------------------------------------------------
# MediaPipe Setup
# ---------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------
# Landmark Indices
# ---------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454

# ---------------------------------------------------
# Fatigue Detection Settings
# ---------------------------------------------------
EAR_THRESHOLD = 0.20
CLOSED_EYES_FRAMES = 15

frame_counter = 0

last_fatigue_state = False
last_left_state = False
last_right_state = False

# ---------------------------------------------------
# Event Logging Setup
# ---------------------------------------------------
LOG_FILE = "results/fatigue_log.csv"

if not os.path.exists(LOG_FILE):

    with open(LOG_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["timestamp", "event"])

# ---------------------------------------------------
# Webcam Setup
# ---------------------------------------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(2)

# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    # ---------------------------------------------------
    # Status Variables
    # ---------------------------------------------------
    fatigue_status = ""
    distraction_status = ""

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ---------------------------------------------------
            # Draw Face Mesh
            # ---------------------------------------------------
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )

            # ---------------------------------------------------
            # Eye Landmark Extraction
            # ---------------------------------------------------
            left_eye_points = []
            right_eye_points = []

            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * frame_width)
                y = int(face_landmarks.landmark[idx].y * frame_height)

                left_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            for idx in RIGHT_EYE:

                x = int(face_landmarks.landmark[idx].x * frame_width)
                y = int(face_landmarks.landmark[idx].y * frame_height)

                right_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # ---------------------------------------------------
            # EAR Calculation
            # ---------------------------------------------------
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)

            avg_ear = (left_ear + right_ear) / 2.0

            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # ---------------------------------------------------
            # Drowsiness Detection
            # ---------------------------------------------------
            if avg_ear < EAR_THRESHOLD:

                frame_counter += 1

                if frame_counter >= CLOSED_EYES_FRAMES:

                    fatigue_status = "DROWSINESS DETECTED"

                    if not last_fatigue_state:

                        log_event("drowsiness")
                        last_fatigue_state = True

            else:
                frame_counter = 0
                last_fatigue_state = False

            # ---------------------------------------------------
            # Head Direction Detection
            # ---------------------------------------------------
            nose_x = int(face_landmarks.landmark[NOSE_TIP].x * frame_width)

            left_x = int(face_landmarks.landmark[LEFT_FACE].x * frame_width)

            right_x = int(face_landmarks.landmark[RIGHT_FACE].x * frame_width)

            face_center = (left_x + right_x) // 2

            direction_offset = nose_x - face_center

            cv2.putText(
                frame,
                f"Offset: {direction_offset}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            # ---------------------------------------------------
            # Distraction Detection
            # ---------------------------------------------------
            if direction_offset > 15:

                distraction_status = "LOOKING RIGHT"

                if not last_right_state:

                    log_event("looking_right")
                    last_right_state = True

                last_left_state = False

            elif direction_offset < -15:

                distraction_status = "LOOKING LEFT"

                if not last_left_state:

                    log_event("looking_left")
                    last_left_state = True

                last_right_state = False

            else:

                last_left_state = False
                last_right_state = False

            # ---------------------------------------------------
            # Display Status Messages
            # ---------------------------------------------------

            # Normal attentive state
            if fatigue_status == "" and distraction_status == "":

                cv2.putText(
                    frame,
                    "STATUS: ATTENTIVE",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

            # ---------------------------------------------------
            # Fatigue Warning
            # ---------------------------------------------------
            if fatigue_status != "":

                threading.Thread(target=play_alarm).start()

                cv2.putText(
                    frame,
                    fatigue_status,
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # ---------------------------------------------------
            # Distraction Warning
            # ---------------------------------------------------
            if distraction_status != "":

                threading.Thread(target=play_alarm).start()

                cv2.putText(
                    frame,
                    distraction_status,
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # ---------------------------------------------------
    # Show Window
    # ---------------------------------------------------
    cv2.imshow("Unified AI Driver Monitoring System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------
# Cleanup
# ---------------------------------------------------
cap.release()
cv2.destroyAllWindows()