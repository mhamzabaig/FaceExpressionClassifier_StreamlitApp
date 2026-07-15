"""Application services — orchestration that wires infrastructure together."""

from src.application.webcam_pipeline import FaceResult, WebcamPipeline

__all__ = [
    "FaceResult",
    "WebcamPipeline",
]
