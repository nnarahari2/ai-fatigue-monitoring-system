import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------
# MediaPipe Setup
# ----------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Key face landmarks
NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454

# ----------------------------------------
# Webcam Setup
# ----------------------------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(2)

# ----------------------------------------
# Main Loop
# ----------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # Draw face mesh
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

            # Get landmark positions
            nose_x = int(face_landmarks.landmark[NOSE_TIP].x * frame_width)

            left_x = int(face_landmarks.landmark[LEFT_FACE].x * frame_width)

            right_x = int(face_landmarks.landmark[RIGHT_FACE].x * frame_width)

            # Head direction estimation
            face_center = (left_x + right_x) // 2

            direction_offset = nose_x - face_center

            # Display direction value
            cv2.putText(
                frame,
                f"Offset: {direction_offset}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Looking right
            if direction_offset > 15:

                cv2.putText(
                    frame,
                    "LOOKING RIGHT",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # Looking left
            elif direction_offset < -15:

                cv2.putText(
                    frame,
                    "LOOKING LEFT",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # Looking forward
            else:

                cv2.putText(
                    frame,
                    "ATTENTIVE",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Distraction Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()