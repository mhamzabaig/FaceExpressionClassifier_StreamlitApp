"""Presentation adapters (WebRTC/Streamlit glue). No inference logic lives here."""

from src.presentation.video_processor import EmotionVideoProcessor, StreamMetrics

__all__ = [
    "EmotionVideoProcessor",
    "StreamMetrics",
]
