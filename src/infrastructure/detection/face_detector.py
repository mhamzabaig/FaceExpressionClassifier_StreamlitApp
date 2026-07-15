"""MediaPipe-based multi-face detector (Tasks API / BlazeFace).

This module wraps the MediaPipe **Tasks** Face Detector behind a small,
framework-agnostic API. It is responsible for *one thing only*: locating faces
in a frame and returning their bounding boxes with detection confidence.

We use the Tasks API (``mediapipe.tasks.python.vision.FaceDetector``) rather than
the legacy ``mediapipe.solutions.face_detection`` API, because the latter is
deprecated and has been removed from current mediapipe releases (0.10.x). The
Tasks API loads a small BlazeFace ``.tflite`` asset bundled in ``assets/models/``.

Explicitly out of scope (by design):
    * No emotion / expression prediction.
    * No Streamlit or any UI code.
    * No drawing / annotation of frames.

The public contract is unchanged::

    detector = FaceDetector()
    detections = detector.detect(frame)   # frame: HxWx3 uint8 ndarray
    # -> [{"bbox": (x1, y1, x2, y2), "confidence": 0.97}, ...]

Bounding-box coordinates are absolute pixels in the input frame's coordinate
space, clamped to the frame bounds, with ``x2 > x1`` and ``y2 > y1`` guaranteed.
"""

from __future__ import annotations

import logging
import os
from types import TracebackType
from typing import List, Optional, Tuple, Type, TypedDict

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError(
        "mediapipe (>=0.10) is required for FaceDetector. Install it via "
        "`pip install mediapipe` (see requirements.txt)."
    ) from exc

logger = logging.getLogger(__name__)

# Default BlazeFace short-range model asset, resolved relative to the repo root
# (src/infrastructure/detection/face_detector.py -> up 3 -> project root).
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
DEFAULT_DETECTOR_MODEL: str = os.path.join(
    _PROJECT_ROOT, "assets", "models", "blaze_face_short_range.tflite"
)


class FaceDetectorError(RuntimeError):
    """Raised when the detector cannot be initialised or has been closed.

    Per-frame inference failures are intentionally *not* raised through this
    exception. To keep a real-time video stream alive, :meth:`FaceDetector.detect`
    logs such failures and returns an empty list instead of propagating them.
    """


class Detection(TypedDict):
    """A single detected face.

    Attributes:
        bbox: Absolute pixel bounding box ``(x1, y1, x2, y2)`` where
            ``(x1, y1)`` is the top-left and ``(x2, y2)`` the bottom-right
            corner. Coordinates are clamped to the frame and satisfy
            ``x2 > x1`` and ``y2 > y1``.
        confidence: Detection confidence score in the range ``[0.0, 1.0]``.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float


class FaceDetector:
    """Detect multiple faces in a frame and return their bounding boxes.

    A single Tasks FaceDetector is created once at construction and reused for
    every :meth:`detect` call, which is essential for real-time throughput.

    The instance is *not* thread-safe: the underlying detector keeps internal
    state. Use one detector per processing thread. In this project a single
    detector lives behind the WebRTC video callback thread.

    Usable as a context manager to guarantee resource release::

        with FaceDetector() as detector:
            detections = detector.detect(frame)

    Attributes:
        min_detection_confidence: Minimum score for a detection to be returned.
        model_asset_path: Path to the BlazeFace ``.tflite`` asset.
        input_is_bgr: Whether incoming frames are BGR (OpenCV convention).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_asset_path: str = DEFAULT_DETECTOR_MODEL,
        input_is_bgr: bool = True,
    ) -> None:
        """Initialise the detector and build the MediaPipe Tasks graph.

        Args:
            min_detection_confidence: Minimum confidence in ``[0.0, 1.0]`` for a
                face to be reported. Defaults to ``0.5``.
            model_asset_path: Path to the BlazeFace short-range ``.tflite`` model.
                Defaults to the asset bundled at
                ``assets/models/blaze_face_short_range.tflite``.
            input_is_bgr: If ``True`` (default), frames passed to :meth:`detect`
                are assumed BGR (OpenCV) and converted to RGB internally, as
                MediaPipe requires RGB. Set ``False`` if you supply RGB frames.

        Raises:
            ValueError: If ``min_detection_confidence`` is outside ``[0, 1]``.
            FileNotFoundError: If ``model_asset_path`` does not exist.
            FaceDetectorError: If the MediaPipe graph fails to initialise.
        """
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(
                "min_detection_confidence must be in [0.0, 1.0], got "
                f"{min_detection_confidence!r}"
            )
        if not os.path.isfile(model_asset_path):
            raise FileNotFoundError(
                f"Face detector model asset not found: {model_asset_path!r}"
            )

        self.min_detection_confidence = min_detection_confidence
        self.model_asset_path = model_asset_path
        self.input_is_bgr = input_is_bgr
        self._closed = False

        try:
            options = mp_vision.FaceDetectorOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_asset_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=min_detection_confidence,
            )
            self._detector = mp_vision.FaceDetector.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - surface any init failure uniformly
            raise FaceDetectorError(
                f"Failed to initialise MediaPipe Tasks FaceDetector: {exc}"
            ) from exc

        logger.info(
            "FaceDetector initialised (Tasks API, min_detection_confidence=%.2f, "
            "model=%s, input_is_bgr=%s)",
            min_detection_confidence,
            os.path.basename(model_asset_path),
            input_is_bgr,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect all faces in a single frame.

        Args:
            frame: An ``(H, W, 3)`` ``uint8`` image. Assumed BGR when
                ``input_is_bgr`` is ``True`` (OpenCV default), otherwise RGB.

        Returns:
            A list of :class:`Detection` dicts, one per detected face. Empty when
            no faces are found or a per-frame inference error is caught (logged,
            not raised).

        Raises:
            ValueError: If ``frame`` is not a 3-channel ``uint8`` ndarray.
            FaceDetectorError: If called after the detector has been closed.
        """
        if self._closed:
            raise FaceDetectorError("detect() called on a closed FaceDetector.")

        self._validate_frame(frame)
        height, width = frame.shape[:2]

        try:
            rgb = self._to_rgb(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)
        except Exception as exc:  # noqa: BLE001 - keep the stream alive on failure
            logger.warning("Face detection failed on frame; returning []: %s", exc)
            return []

        detections: List[Detection] = []
        for raw in getattr(result, "detections", None) or []:
            detection = self._to_detection(raw, width, height)
            if detection is not None:
                detections.append(detection)

        logger.debug("Detected %d face(s) in frame (%dx%d).", len(detections), width, height)
        return detections

    def _to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Return a contiguous RGB copy of ``frame`` for MediaPipe.

        Args:
            frame: Validated ``(H, W, 3)`` ``uint8`` image.

        Returns:
            A C-contiguous RGB ``uint8`` array (MediaPipe requires contiguity).
        """
        if self.input_is_bgr:
            return np.ascontiguousarray(frame[:, :, ::-1])
        return np.ascontiguousarray(frame)

    @staticmethod
    def _to_detection(raw: object, width: int, height: int) -> Optional[Detection]:
        """Convert one Tasks detection into a :class:`Detection`.

        The Tasks API reports the bounding box in absolute pixels already
        (``origin_x``, ``origin_y``, ``width``, ``height``). Coordinates are
        clamped to the frame; degenerate boxes are dropped.

        Args:
            raw: A single Tasks detection object.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            A :class:`Detection`, or ``None`` if the box is degenerate/malformed.
        """
        try:
            box = raw.bounding_box  # type: ignore[attr-defined]
            confidence = float(raw.categories[0].score)  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("Skipping malformed MediaPipe detection: %s", exc)
            return None

        x1 = int(box.origin_x)
        y1 = int(box.origin_y)
        x2 = x1 + int(box.width)
        y2 = y1 + int(box.height)

        # Clamp to frame bounds; BlazeFace can return coords slightly off-frame.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            logger.debug("Dropping degenerate bbox (%d, %d, %d, %d).", x1, y1, x2, y2)
            return None

        return Detection(bbox=(x1, y1, x2, y2), confidence=confidence)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate that ``frame`` is a 3-channel ``uint8`` image.

        Args:
            frame: The object passed to :meth:`detect`.

        Raises:
            ValueError: If ``frame`` is not an ``(H, W, 3)`` ``uint8`` ndarray.
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"frame must be a numpy.ndarray, got {type(frame).__name__}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must have shape (H, W, 3), got {frame.shape!r}.")
        if frame.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got dtype {frame.dtype!r}.")

    def close(self) -> None:
        """Release the underlying detector. Safe to call multiple times."""
        # getattr guards against a failure during __init__ (before attrs exist).
        if getattr(self, "_closed", False):
            return
        detector = getattr(self, "_detector", None)
        if detector is not None:
            try:
                detector.close()
            except Exception as exc:  # noqa: BLE001 - closing must never raise
                logger.warning("Error while closing FaceDetector: %s", exc)
        self._closed = True
        logger.info("FaceDetector closed.")

    def __enter__(self) -> "FaceDetector":
        """Enter the runtime context and return the detector."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the runtime context, releasing MediaPipe resources."""
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort finaliser
        """Best-effort resource release if :meth:`close` was not called."""
        self.close()
