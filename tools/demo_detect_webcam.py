"""Local real-time smoke-test for the full flow so far (development only).

Pipeline exercised: webcam -> FaceDetector -> FacePreprocessor -> EmotionClassifier.
Draws bounding boxes, the predicted emotion + confidence, and live FPS in a
preview window. This is NOT part of the deployed app (the Cloud app uses
streamlit-webrtc); it exists purely to verify the flow end-to-end on your machine.

Run from the project root:

    python tools/demo_detect_webcam.py

The model is chosen by ``MODEL_NAME`` in ``src/config.py`` (its extension picks
the TFLite or Keras backend automatically). You can still override the resolved
path with the env var MODEL_PATH. Press 'q' to quit.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import cv2  # opencv-python (GUI build) — fine for a local preview window
import numpy as np

# Make the project root importable so `src...` resolves when run as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import MODEL_INPUT_SIZE, get_model_path  # noqa: E402
from src.infrastructure.detection import FaceDetector  # noqa: E402
from src.infrastructure.inference import create_classifier  # noqa: E402
from src.infrastructure.preprocessing import (  # noqa: E402
    FacePreprocessor,
    NormalizationMode,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CAMERA_INDEX = 0
# Resolve the model chosen in src/config.py (env var still wins if set).
MODEL_PATH = os.environ.get("MODEL_PATH", get_model_path())
BOX_COLOR = (0, 255, 0)      # BGR — green
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 128, 0)


def main() -> int:
    """Run the live detect -> classify preview loop.

    Returns:
        Process exit code: 0 on clean exit, 1 on camera/model failure.
    """
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_INDEX}.")
        return 1

    # This model expects RAW [0,255] pixels — it has its own rescaling layer
    # baked in. Feeding pre-scaled [0,1]/[-1,1] froze output at "Sad ~0.92";
    # raw pixels make it input-sensitive (verified via normalization probe).
    preprocessor = FacePreprocessor(normalization=NormalizationMode.NONE)

    fps = 0.0
    prev = time.time()

    with FaceDetector(min_detection_confidence=0.5) as detector, \
            create_classifier(MODEL_PATH, input_size=MODEL_INPUT_SIZE) as classifier:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: failed to read frame; stopping.")
                break

            for det in detector.detect(frame):  # frame is BGR
                x1, y1, x2, y2 = det["bbox"]
                try:
                    tensor = preprocessor.preprocess(frame, det["bbox"])
                    result = classifier.predict(tensor)
                    caption = f"{result['label']} {result['confidence']:.2f}"
                except Exception as exc:  # keep the stream alive per-face
                    caption = "?"
                    logging.warning("per-face inference failed: %s", exc)

                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                _draw_label(frame, caption, x1, y1)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(
                frame,
                f"fps: {fps:4.1f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Emotion demo (press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def _draw_label(frame: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw a filled caption box with the emotion text above a face box."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = max(0, y - th - base - 6)
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + base + 6), TEXT_BG, -1)
    cv2.putText(frame, text, (x + 3, top + th + 3), font, scale, TEXT_COLOR, thick)


if __name__ == "__main__":
    raise SystemExit(main())
