"""Face detection infrastructure (MediaPipe BlazeFace)."""

from src.infrastructure.detection.face_detector import (
    Detection,
    FaceDetector,
    FaceDetectorError,
)

__all__ = ["Detection", "FaceDetector", "FaceDetectorError"]
