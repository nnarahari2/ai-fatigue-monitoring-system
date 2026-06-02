import cv2
import os

cap = cv2.VideoCapture(0)

label = input(
    "Enter label (alert, drowsy, looking_left, looking_right): "
)

save_dir = f"dataset/{label}"

os.makedirs(save_dir, exist_ok=True)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.putText(
        frame,
        f"Images Saved: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):

        filename = os.path.join(
            save_dir,
            f"{label}_{count}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()