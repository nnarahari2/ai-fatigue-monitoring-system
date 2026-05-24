import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

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
# MediaPipe Face Mesh Setup
# ---------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------
# Eye Landmark Indices
# ---------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------
# Drowsiness Detection Settings
# ---------------------------------------------------
EAR_THRESHOLD = 0.20
CLOSED_EYES_FRAMES = 15

frame_counter = 0

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

    # Flip webcam horizontally
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    # ---------------------------------------------------
    # Face Detected
    # ---------------------------------------------------
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

            left_eye_points = []
            right_eye_points = []

            # ---------------------------------------------------
            # LEFT EYE LANDMARKS
            # ---------------------------------------------------
            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * frame_width)
                y = int(face_landmarks.landmark[idx].y * frame_height)

                left_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # ---------------------------------------------------
            # RIGHT EYE LANDMARKS
            # ---------------------------------------------------
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

            # ---------------------------------------------------
            # Display EAR
            # ---------------------------------------------------
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

                    cv2.putText(
                        frame,
                        "DROWSINESS DETECTED",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

            else:
                frame_counter = 0

    # ---------------------------------------------------
    # Show Window
    # ---------------------------------------------------
    cv2.imshow("AI Fatigue Monitoring System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------
# Cleanup
# ---------------------------------------------------
cap.release()
cv2.destroyAllWindows()