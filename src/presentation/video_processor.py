"""WebRTC video processor — the bridge between ``streamlit-webrtc`` and the
inference pipeline.

This module contains **no Streamlit code** on purpose: it converts incoming
``av.VideoFrame`` objects to NumPy arrays, runs :class:`WebcamPipeline`, and
publishes thread-safe metrics (FPS, face count, per-face results) that the UI
polls. It runs entirely on the WebRTC worker thread.

Why a class (not a plain ``video_frame_callback``): ``streamlit-webrtc`` exposes
the factory-built instance as ``ctx.video_processor``, so the Streamlit script —
running on a *different* thread — can read :meth:`EmotionVideoProcessor.get_metrics`
while ``recv`` keeps writing them. All shared state is guarded by a lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import av
import numpy as np

from src.application import FaceResult, WebcamPipeline
from src.infrastructure.preprocessing import NormalizationMode

logger = logging.getLogger(__name__)

#: EMA smoothing factor for the FPS estimate (higher = smoother, slower to react).
_FPS_SMOOTHING: float = 0.9


@dataclass
class StreamMetrics:
    """An immutable snapshot of the stream's current state.

    A fresh instance is published on every frame, so a reader that grabs one
    under the lock gets a consistent view and can then release the lock.

    Attributes:
        fps: Smoothed frames-per-second of the processing loop.
        face_count: Number of faces detected in the latest frame.
        results: Per-face results for the latest frame.
    """

    fps: float = 0.0
    face_count: int = 0
    results: List[FaceResult] = field(default_factory=list)


class EmotionVideoProcessor:
    """Per-connection WebRTC processor: frame in, annotated frame out.

    One instance is created by ``streamlit-webrtc`` per stream start (via the
    factory), so the model is loaded once per session rather than per frame.

    ``recv`` runs on the WebRTC worker thread; :meth:`get_metrics` is called from
    the Streamlit script thread. The lock makes that hand-off safe.
    """

    def __init__(
        self,
        normalization: NormalizationMode = NormalizationMode.NONE,
        min_detection_confidence: float = 0.5,
        max_faces: Optional[int] = None,
    ) -> None:
        """Build the pipeline and initialise metric state.

        Args:
            normalization: Pixel normalization for the classifier. Defaults to
                :data:`NormalizationMode.NONE` (raw ``[0, 255]``), matching the
                current model whose rescaling is baked in.
            min_detection_confidence: Minimum face-detection score in ``[0, 1]``.
            max_faces: Optional per-frame cap on classified faces (largest first).
        """
        self._pipeline = WebcamPipeline.from_config(
            min_detection_confidence=min_detection_confidence,
            normalization=normalization,
            max_faces=max_faces,
        )
        self._lock = threading.Lock()
        self._fps = 0.0
        self._prev_t: Optional[float] = None
        self._metrics = StreamMetrics()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one WebRTC frame and return the annotated frame.

        Never raises into the WebRTC loop: on failure it logs and returns the
        original frame so the stream stays alive.

        Args:
            frame: The incoming video frame from the browser.

        Returns:
            An ``av.VideoFrame`` (BGR) with boxes + labels drawn, or the original
            frame if processing failed.
        """
        image = frame.to_ndarray(format="bgr24")
        try:
            annotated, results = self._pipeline.process_with_results(image)
        except Exception as exc:  # noqa: BLE001 - keep the stream alive
            logger.warning("Frame processing failed; passing frame through: %s", exc)
            return frame

        self._update_metrics(results)

        new_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def get_metrics(self) -> StreamMetrics:
        """Return the latest metrics snapshot (thread-safe).

        Returns:
            The most recently published :class:`StreamMetrics`.
        """
        with self._lock:
            return self._metrics

    def _update_metrics(self, results: List[FaceResult]) -> None:
        """Update the smoothed FPS and publish a new metrics snapshot.

        Args:
            results: The latest frame's per-face results.
        """
        now = time.monotonic()
        with self._lock:
            if self._prev_t is not None:
                dt = now - self._prev_t
                if dt > 0:
                    instant = 1.0 / dt
                    self._fps = (
                        instant
                        if self._fps == 0.0
                        else _FPS_SMOOTHING * self._fps + (1.0 - _FPS_SMOOTHING) * instant
                    )
            self._prev_t = now
            self._metrics = StreamMetrics(
                fps=round(self._fps, 1),
                face_count=len(results),
                results=results,
            )

    def close(self) -> None:
        """Release the underlying pipeline. Safe to call multiple times."""
        try:
            self._pipeline.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            logger.warning("Error closing pipeline: %s", exc)

    def __del__(self) -> None:  # pragma: no cover - best-effort finaliser
        """Best-effort release if the stream ended without an explicit close."""
        self.close()
