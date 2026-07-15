"""Frame-in / frame-out webcam emotion pipeline (no UI-framework knowledge).

Wires the four infrastructure pieces into a single per-frame call::

    webcam frame (BGR)
        -> MediaPipe face detection
        -> for each face: crop -> resize/normalize -> emotion classification
        -> draw boxes + labels
        -> annotated frame (BGR)

The pipeline is deliberately ignorant of Streamlit / WebRTC / OpenCV windows:
:meth:`WebcamPipeline.process` takes a NumPy BGR frame and returns a NumPy BGR
frame. Whoever owns the transport (a ``streamlit-webrtc`` callback, the local
``cv2`` demo, a unit test) adapts to/from that contract.

Performance notes (the goal is maximum FPS):
    * Detector, preprocessor, classifier and drawer are built once and reused;
      the classifier is warmed up at construction.
    * One detection pass per frame; empty frames short-circuit immediately.
    * Annotation is done in place — the input frame is only copied when it is
      not writeable (e.g. a read-only WebRTC buffer).
    * ``max_faces`` optionally caps how many faces are classified per frame
      (largest boxes first), bounding worst-case cost on crowded frames.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import List, Optional, Tuple, Type, TypedDict

import numpy as np

from src.config import MODEL_INPUT_SIZE, get_model_path
from src.infrastructure.detection import FaceDetector
from src.infrastructure.inference import create_classifier
from src.infrastructure.inference.emotion_classifier import _BaseEmotionClassifier
from src.infrastructure.preprocessing import FacePreprocessor, NormalizationMode
from src.infrastructure.rendering import EmotionDrawer

logger = logging.getLogger(__name__)


class FaceResult(TypedDict):
    """One face's detection + classification result.

    Attributes:
        bbox: Absolute-pixel ``(x1, y1, x2, y2)`` box.
        class_id: Predicted class id, or ``-1`` if classification failed.
        label: Emotion label, or ``"?"`` if classification failed.
        confidence: Predicted-class probability in ``[0, 1]`` (``0.0`` on failure).
    """

    bbox: Tuple[int, int, int, int]
    class_id: int
    label: str
    confidence: float


class WebcamPipeline:
    """Detect faces, classify emotions and annotate a frame, end to end.

    Not thread-safe as a whole (the MediaPipe detector keeps internal state):
    use one pipeline per processing thread. In the deployed app a single
    pipeline lives behind the WebRTC video callback thread.

    Attributes:
        annotate: Whether :meth:`process` draws results onto the frame.
        max_faces: Optional cap on faces classified per frame (largest first).
    """

    def __init__(
        self,
        detector: FaceDetector,
        preprocessor: FacePreprocessor,
        classifier: _BaseEmotionClassifier,
        drawer: EmotionDrawer,
        annotate: bool = True,
        max_faces: Optional[int] = None,
        own_classifier: bool = True,
    ) -> None:
        """Assemble a pipeline from ready-made components (dependency injection).

        Use :meth:`from_config` for the common case of building everything from
        :mod:`src.config`; this constructor is for tests and custom wiring.

        Args:
            detector: Configured :class:`FaceDetector`.
            preprocessor: Configured :class:`FacePreprocessor` (its ``target_size``
                must match the classifier's expected input).
            classifier: A classifier from :func:`create_classifier`.
            drawer: Configured :class:`EmotionDrawer`.
            annotate: If ``True`` (default), :meth:`process` draws onto the frame.
            max_faces: If set, classify only the ``max_faces`` largest faces per
                frame. ``None`` (default) classifies every detected face.
            own_classifier: If ``True`` (default), :meth:`close` also closes the
                classifier. Set ``False`` when the classifier is shared/cached
                (e.g. via ``st.cache_resource``) and must outlive this pipeline.

        Raises:
            ValueError: If ``max_faces`` is not positive.
        """
        if max_faces is not None and max_faces <= 0:
            raise ValueError(f"max_faces must be positive or None, got {max_faces!r}.")

        self._detector = detector
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._drawer = drawer
        self.annotate = annotate
        self.max_faces = max_faces
        self._own_classifier = own_classifier
        self._closed = False

    @classmethod
    def from_config(
        cls,
        min_detection_confidence: float = 0.5,
        normalization: NormalizationMode = NormalizationMode.NONE,
        margin: float = 0.0,
        annotate: bool = True,
        max_faces: Optional[int] = None,
        classifier: Optional[_BaseEmotionClassifier] = None,
    ) -> "WebcamPipeline":
        """Build a pipeline from :mod:`src.config` and sensible defaults.

        The model is resolved via :func:`src.config.get_model_path` and the
        backend (TFLite/Keras) is chosen from its extension. Input size comes
        from :data:`src.config.MODEL_INPUT_SIZE`, keeping the preprocessor and
        classifier in lockstep.

        Args:
            min_detection_confidence: Minimum face-detection score in ``[0, 1]``.
            normalization: Pixel normalization for the classifier. Defaults to
                :data:`NormalizationMode.NONE` (raw ``[0, 255]``), which matches
                the current model (its rescaling is baked in). Switch to
                ``RESCALE`` if a model expects pre-normalized input.
            margin: Fractional context margin around each crop.
            annotate: Draw results onto the frame. Defaults to ``True``.
            max_faces: Optional per-frame cap (largest faces first).
            classifier: An already-built (typically cached/shared) classifier to
                reuse instead of loading the model again. When provided, the
                pipeline does **not** own it — :meth:`close` leaves it open so it
                can be reused across stream restarts. ``None`` (default) loads the
                model here and owns it.

        Returns:
            A ready-to-use, warmed-up :class:`WebcamPipeline`.
        """
        detector = FaceDetector(min_detection_confidence=min_detection_confidence)
        preprocessor = FacePreprocessor(
            target_size=MODEL_INPUT_SIZE,
            normalization=normalization,
            margin=margin,
        )
        own_classifier = classifier is None
        if classifier is None:
            classifier = create_classifier(get_model_path(), input_size=MODEL_INPUT_SIZE)
        drawer = EmotionDrawer()
        return cls(
            detector=detector,
            preprocessor=preprocessor,
            classifier=classifier,
            drawer=drawer,
            annotate=annotate,
            max_faces=max_faces,
            own_classifier=own_classifier,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run the full pipeline on one frame and return the annotated frame.

        Args:
            frame: An ``(H, W, 3)`` ``uint8`` **BGR** frame.

        Returns:
            The annotated BGR frame. When :attr:`annotate` is ``False`` the frame
            is returned unchanged (use :meth:`predict` for the results instead).

        Raises:
            RuntimeError: If the pipeline has been closed.
            ValueError: If ``frame`` is not a 3-channel ``uint8`` ndarray.
        """
        annotated, _ = self.process_with_results(frame)
        return annotated

    def process_with_results(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[FaceResult]]:
        """Annotate a frame **and** return the per-face results in one pass.

        Detection + classification run only once; this is the method the WebRTC
        layer uses so it can render the frame while also reading face count /
        labels for the UI without re-doing inference.

        Args:
            frame: An ``(H, W, 3)`` ``uint8`` **BGR** frame.

        Returns:
            A ``(annotated_frame, results)`` tuple. ``results`` is empty when no
            faces are detected. The frame is annotated only when :attr:`annotate`
            is ``True``.

        Raises:
            RuntimeError: If the pipeline has been closed.
            ValueError: If ``frame`` is not a 3-channel ``uint8`` ndarray.
        """
        self._ensure_open()

        # Annotate in place, but never mutate a read-only buffer (e.g. WebRTC).
        if isinstance(frame, np.ndarray) and not frame.flags.writeable:
            frame = frame.copy()

        results = self.predict(frame)
        if self.annotate and results:
            for res in results:
                confidence = res["confidence"] if res["class_id"] >= 0 else None
                self._drawer.draw(frame, res["bbox"], res["label"], confidence)
        return frame, results

    def predict(self, frame: np.ndarray) -> List[FaceResult]:
        """Detect and classify faces in ``frame`` without drawing.

        Useful for callers that want the structured results (logging, metrics,
        custom rendering) rather than an annotated frame.

        Args:
            frame: An ``(H, W, 3)`` ``uint8`` **BGR** frame.

        Returns:
            One :class:`FaceResult` per face (empty if none detected). A face
            whose classification raises still yields a result with ``class_id``
            ``-1`` / ``label`` ``"?"`` so its box can still be drawn.

        Raises:
            EmotionClassifierError: If the pipeline has been closed.
            ValueError: If ``frame`` is not a 3-channel ``uint8`` ndarray.
        """
        self._ensure_open()

        detections = self._detector.detect(frame)
        if not detections:
            return []

        if self.max_faces is not None and len(detections) > self.max_faces:
            detections = sorted(detections, key=self._box_area, reverse=True)[: self.max_faces]

        results: List[FaceResult] = []
        for det in detections:
            bbox = det["bbox"]
            try:
                tensor = self._preprocessor.preprocess(frame, bbox)
                pred = self._classifier.predict(tensor)
            except Exception as exc:  # noqa: BLE001 - keep the stream alive per-face
                logger.warning("Per-face inference failed; box kept as '?': %s", exc)
                results.append(
                    FaceResult(bbox=bbox, class_id=-1, label="?", confidence=0.0)
                )
                continue
            results.append(
                FaceResult(
                    bbox=bbox,
                    class_id=pred["class_id"],
                    label=pred["label"],
                    confidence=pred["confidence"],
                )
            )
        return results

    @staticmethod
    def _box_area(detection: dict) -> int:
        """Return the pixel area of a detection's bbox (for ``max_faces`` ranking).

        Args:
            detection: A detector result with a ``"bbox"`` ``(x1, y1, x2, y2)``.

        Returns:
            The box area in pixels.
        """
        x1, y1, x2, y2 = detection["bbox"]
        return (x2 - x1) * (y2 - y1)

    def _ensure_open(self) -> None:
        """Raise if the pipeline has been closed.

        Raises:
            RuntimeError: If :meth:`close` has been called.
        """
        if self._closed:
            raise RuntimeError("process()/predict() called on a closed WebcamPipeline.")

    def close(self) -> None:
        """Release the detector (and the classifier if owned). Safe to call twice.

        A classifier injected via ``from_config(classifier=...)`` is left open so
        a shared/cached instance survives for the next stream.
        """
        if self._closed:
            return
        self._closed = True
        self._detector.close()
        if self._own_classifier:
            self._classifier.close()
        logger.info("WebcamPipeline closed (classifier owned=%s).", self._own_classifier)

    def __enter__(self) -> "WebcamPipeline":
        """Enter the runtime context and return the pipeline."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the runtime context, releasing resources."""
        self.close()
