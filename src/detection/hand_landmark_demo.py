"""Phase 1 demo: detect and draw hand landmarks from the webcam.

Run from the project root:
    python src/detection/hand_landmark_demo.py

Press q to close the webcam window.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import mediapipe as mp


MODEL_PATH = Path("models/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape

    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(frame, points[start_index], points[end_index], (0, 255, 0), 2)

    for index, point in enumerate(points):
        color = (0, 0, 255) if index in [4, 8, 12, 16, 20] else (255, 255, 255)
        cv2.circle(frame, point, 5, color, -1)


def main():
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        print("Download hand_landmarker.task into the models folder and try again.")
        return

    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Check camera permission and try again.")
        return

    previous_time = time.time()
    printed_sample = False

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

                if not printed_sample:
                    wrist = result.hand_landmarks[0][0]
                    index_tip = result.hand_landmarks[0][8]
                    print("Sample landmark coordinates:")
                    print(f"  Wrist: x={wrist.x:.3f}, y={wrist.y:.3f}, z={wrist.z:.3f}")
                    print(
                        "  Index fingertip: "
                        f"x={index_tip.x:.3f}, y={index_tip.y:.3f}, z={index_tip.z:.3f}"
                    )
                    printed_sample = True

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press q to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Phase 1 - MediaPipe Hand Landmarks", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
