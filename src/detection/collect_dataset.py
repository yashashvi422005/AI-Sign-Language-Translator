"""Phase 2: collect raw sign-language videos from the webcam.

Run from the project root:
    python src/detection/collect_dataset.py --label HELLO --samples 10 --seconds 3

Controls:
    space - start recording the next sample
    q     - quit
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.vocabulary import INITIAL_VOCABULARY, normalize_label


DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Record webcam videos for one sign label.")
    parser.add_argument(
        "--label",
        required=True,
        help="Sign label to record, for example HELLO or THANK_YOU.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of videos to record for this label.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Duration of each video sample in seconds.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Use 0 for the default webcam.",
    )
    return parser.parse_args()


def draw_text(frame, text, y, color=(255, 255, 255), scale=0.8):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def build_output_path(label, sample_number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{sample_number:03d}_{timestamp}.mp4"
    return DATASET_ROOT / label / filename


def record_sample(cap, output_path, label, sample_number, total_samples, seconds):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, DEFAULT_FPS, (width, height))

    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read frame while recording.")
            break

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_time
        remaining = max(0, seconds - elapsed)

        writer.write(frame)

        draw_text(frame, f"Recording {label}", 40, (0, 0, 255), 1.0)
        draw_text(frame, f"Sample {sample_number}/{total_samples}", 80)
        draw_text(frame, f"Time left: {remaining:.1f}s", 120)
        cv2.imshow("Phase 2 - Dataset Collection", frame)

        if elapsed >= seconds:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            writer.release()
            return False

    writer.release()
    print(f"Saved: {output_path}")
    return True


def main():
    args = parse_args()
    label = normalize_label(args.label)

    if label not in INITIAL_VOCABULARY:
        print(f"Unknown label: {label}")
        print("Use one of:")
        for vocabulary_label in INITIAL_VOCABULARY:
            print(f"  - {vocabulary_label}")
        return

    label_dir = DATASET_ROOT / label
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permission and try again.")
        return

    print(f"Collecting {args.samples} samples for: {label}")
    print("Press space to record each sample. Press q to quit.")

    sample_number = 1

    while sample_number <= args.samples:
        success, frame = cap.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        draw_text(frame, f"Label: {label}", 40, (0, 255, 0), 1.0)
        draw_text(frame, f"Next sample: {sample_number}/{args.samples}", 80)
        draw_text(frame, "Press space to record. Press q to quit.", 120)
        cv2.imshow("Phase 2 - Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            output_path = build_output_path(label, sample_number)
            should_continue = record_sample(
                cap,
                output_path,
                label,
                sample_number,
                args.samples,
                args.seconds,
            )
            if not should_continue:
                break
            sample_number += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Dataset collection finished.")


if __name__ == "__main__":
    main()
