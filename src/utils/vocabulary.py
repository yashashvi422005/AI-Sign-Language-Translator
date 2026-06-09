"""Shared vocabulary for the first sign language dataset."""

INITIAL_VOCABULARY = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "HELP",
    "WATER",
    "PLEASE",
    "GOODBYE",
]


def normalize_label(label):
    """Convert user input like 'thank you' into the project label format."""
    return label.strip().upper().replace(" ", "_")

