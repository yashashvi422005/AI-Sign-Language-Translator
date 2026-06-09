# Phase 2: Dataset Creation

## Goal

Create a clean raw video dataset for complete sign words.

Initial labels:

- `HELLO`
- `THANK_YOU`
- `YES`
- `NO`
- `HELP`
- `WATER`
- `PLEASE`
- `GOODBYE`

## Folder Structure

Raw videos are stored like this:

```text
dataset/raw/HELLO/
dataset/raw/THANK_YOU/
dataset/raw/YES/
dataset/raw/NO/
dataset/raw/HELP/
dataset/raw/WATER/
dataset/raw/PLEASE/
dataset/raw/GOODBYE/
```

## Collection Rules

- Record each sign for the same duration.
- Keep your whole signing hand visible.
- Use good lighting.
- Keep the camera steady.
- Avoid a messy background when possible.
- Record from a similar distance each time.
- Add natural variation: slight speed changes, hand position changes, and small angle changes.
- Do not mix two different meanings inside one label folder.

## Gesture Variations

If two gestures mean the same English word, store them under the same label.

Example:

```text
dataset/raw/HELLO/
```

can contain multiple natural ways of signing `HELLO`.

This teaches the model that different-looking gestures can still mean the same output word. Do not create `HELLO_1` and `HELLO_2` unless you want the model to treat them as different classes.

## Recommended First Test

Start small before collecting the full dataset:

```bash
python src/detection/collect_dataset.py --label HELLO --samples 3 --seconds 3
```

Controls:

- Press `space` to record a sample.
- Press `q` to quit.
