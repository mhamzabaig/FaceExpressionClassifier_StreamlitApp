"""Emotion inference infrastructure (TensorFlow Lite or Keras).

Use :func:`create_classifier` — it picks the backend from the model file's
extension. The concrete backend classes are exported for type hints/tests.
"""

from src.infrastructure.inference.emotion_classifier import (
    DEFAULT_LABELS,
    KERAS_EXTENSIONS,
    TFLITE_EXTENSIONS,
    EmotionClassifierError,
    EmotionResult,
    KerasEmotionClassifier,
    TFLiteEmotionClassifier,
    create_classifier,
)

__all__ = [
    "DEFAULT_LABELS",
    "KERAS_EXTENSIONS",
    "TFLITE_EXTENSIONS",
    "EmotionClassifierError",
    "EmotionResult",
    "KerasEmotionClassifier",
    "TFLiteEmotionClassifier",
    "create_classifier",
]
