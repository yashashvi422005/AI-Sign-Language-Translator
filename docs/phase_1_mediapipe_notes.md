# Phase 1: MediaPipe Hand Landmark Notes

## Goal

Open the webcam, detect hands, draw hand landmarks, and understand what MediaPipe returns.

## Key Idea

MediaPipe Hands returns 21 landmarks for each detected hand.

This project uses the newer MediaPipe Tasks API, which requires a model file:

```text
models/hand_landmarker.task
```

Each landmark has:

- `x`: horizontal position, normalized from 0 to 1
- `y`: vertical position, normalized from 0 to 1
- `z`: relative depth value

Because `x` and `y` are normalized, they are not pixel coordinates yet. For example, if the frame width is 1280 and a landmark has `x = 0.5`, its pixel x-position is around `640`.

## Important Landmark IDs

- `0`: wrist
- `4`: thumb tip
- `8`: index finger tip
- `12`: middle finger tip
- `16`: ring finger tip
- `20`: pinky finger tip

## Run Command

```bash
python src/detection/hand_landmark_demo.py
```

Press `q` to close the webcam window.
