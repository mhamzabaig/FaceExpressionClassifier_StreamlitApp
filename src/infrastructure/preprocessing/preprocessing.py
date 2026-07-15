"""Face preprocessing pipeline (detection bbox -> inference-ready tensor).

Transforms a raw camera frame + a detected bounding box into the exact array
layout the classifier expects. Responsibilities, in order:

    1. Crop the face region (with optional context margin).
    2. Handle boundary conditions (clamp boxes that fall outside the frame).
    3. Resize to the model input size (default 256x256).
    4. Convert BGR -> RGB (OpenCV frames are BGR; training used RGB).
    5. Normalize to match TensorFlow training.
    6. Expand dims -> add the batch axis.
    7. Return a NumPy array ready for inference.

Explicitly out of scope:
    * No model loading and no int8 quantization. Quantization needs the model's
      input tensor scale/zero-point, which lives in the classifier. This module
      emits the float32 tensor exactly as the training pipeline fed it.

.. warning::
    The normalization scheme MUST match what the model was trained with. We do
    not have the training script, so :data:`NormalizationMode.RESCALE`
    (``x / 255`` -> ``[0, 1]``) is a *default assumption*, not a known fact.
    If predictions look random/biased toward one class, this is the first thing
    to change (try :data:`NormalizationMode.SYMMETRIC`). Verify against training.

The functions are pure and reusable on their own; :class:`FacePreprocessor`
bundles a fixed configuration for the hot path.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Default model input size as (height, width).
DEFAULT_INPUT_SIZE: Tuple[int, int] = (256, 256)


class NormalizationMode(Enum):
    """Pixel normalization strategy.

    Attributes:
        RESCALE: ``x / 255.0`` -> ``[0.0, 1.0]``. Matches ``keras.layers.Rescaling(1./255)``.
        SYMMETRIC: ``x / 127.5 - 1.0`` -> ``[-1.0, 1.0]``. Matches MobileNet/Inception-style
            ``preprocess_input``.
        NONE: Cast to float32 in ``[0.0, 255.0]`` with no scaling (e.g. EfficientNet,
            which normalizes internally).
    """

    RESCALE = "rescale"
    SYMMETRIC = "symmetric"
    NONE = "none"


def crop_face(
    image: np.ndarray,
    bbox: Sequence[int],
    margin: float = 0.0,
) -> np.ndarray:
    """Crop the face region from ``image``, clamped to the frame bounds.

    Args:
        image: Source frame, ``(H, W, 3)`` ``uint8``.
        bbox: ``(x1, y1, x2, y2)`` absolute-pixel box (as returned by the detector).
        margin: Fractional padding added around the box on each side, expressed as
            a fraction of the box's width/height (e.g. ``0.2`` = +20%). Useful when
            training crops included surrounding context. Clamped to the frame, so
            the effective margin shrinks near edges. Defaults to ``0.0``.

    Returns:
        The cropped BGR region, ``(h, w, 3)`` ``uint8``.

    Raises:
        ValueError: If ``bbox`` is not 4 values or the crop is empty after clamping.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values (x1, y1, x2, y2), got {bbox!r}.")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)

    if margin > 0.0:
        box_w = x2 - x1
        box_h = y2 - y1
        x1 -= margin * box_w
        x2 += margin * box_w
        y1 -= margin * box_h
        y2 += margin * box_h

    # Boundary handling: clamp to the frame so an oversized/edge box stays valid.
    xi1 = max(0, min(int(round(x1)), width - 1))
    yi1 = max(0, min(int(round(y1)), height - 1))
    xi2 = max(0, min(int(round(x2)), width))
    yi2 = max(0, min(int(round(y2)), height))

    if xi2 <= xi1 or yi2 <= yi1:
        raise ValueError(
            f"Empty crop after clamping bbox {tuple(bbox)} to frame {width}x{height}."
        )

    return image[yi1:yi2, xi1:xi2]


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    preserve_aspect_ratio: bool = False,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize ``image`` to ``size``.

    Args:
        image: Image to resize, ``(h, w, 3)``.
        size: Target size as ``(height, width)``. Defaults to :data:`DEFAULT_INPUT_SIZE`.
        preserve_aspect_ratio: If ``True``, scale to fit inside ``size`` and pad the
            remainder (letterbox) so faces are not distorted. If ``False`` (default),
            resize directly to ``size`` (may distort aspect ratio). Whichever matches
            training is what you should use.
        pad_value: Border fill value used when ``preserve_aspect_ratio`` is ``True``.

    Returns:
        The resized image, exactly ``(size[0], size[1], 3)``.
    """
    target_h, target_w = size
    src_h, src_w = image.shape[:2]

    if not preserve_aspect_ratio:
        interp = _pick_interpolation(src_w * src_h, target_w * target_h)
        return cv2.resize(image, (target_w, target_h), interpolation=interp)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = _pick_interpolation(src_w * src_h, new_w * new_h)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB.

    Args:
        image: ``(H, W, 3)`` BGR image (OpenCV convention).

    Returns:
        The image in RGB channel order.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_image(
    image: np.ndarray,
    mode: NormalizationMode = NormalizationMode.RESCALE,
) -> np.ndarray:
    """Normalize pixel values to match TensorFlow training.

    Args:
        image: ``uint8`` image with values in ``[0, 255]``.
        mode: Normalization strategy. See :class:`NormalizationMode`.

    Returns:
        A ``float32`` array with the same shape as ``image``.

    Raises:
        ValueError: If ``mode`` is not a recognised :class:`NormalizationMode`.
    """
    array = image.astype(np.float32)
    if mode is NormalizationMode.RESCALE:
        return array / 255.0
    if mode is NormalizationMode.SYMMETRIC:
        return array / 127.5 - 1.0
    if mode is NormalizationMode.NONE:
        return array
    raise ValueError(f"Unknown normalization mode: {mode!r}.")


def add_batch_dim(array: np.ndarray) -> np.ndarray:
    """Prepend a batch axis: ``(H, W, C)`` -> ``(1, H, W, C)``.

    Args:
        array: The single-image tensor.

    Returns:
        The array with a leading batch dimension of size 1.
    """
    return np.expand_dims(array, axis=0)


def _pick_interpolation(src_area: int, dst_area: int) -> int:
    """Choose an OpenCV interpolation flag based on scaling direction.

    ``INTER_AREA`` is best for shrinking; ``INTER_LINEAR`` (bilinear, matching
    TensorFlow's default ``tf.image.resize``) for enlarging.

    Args:
        src_area: Source pixel area.
        dst_area: Destination pixel area.

    Returns:
        An OpenCV interpolation flag.
    """
    return cv2.INTER_AREA if dst_area < src_area else cv2.INTER_LINEAR


class FacePreprocessor:
    """Reusable, configured face-preprocessing pipeline.

    Bundles a fixed configuration so the per-frame hot path is a single call.
    Stateless and thread-safe (no mutable state between calls).

    Attributes:
        target_size: Model input size as ``(height, width)``.
        normalization: Pixel normalization strategy.
        margin: Fractional context margin added around each crop.
        input_is_bgr: Whether input frames are BGR (OpenCV) and need RGB conversion.
        preserve_aspect_ratio: Whether to letterbox instead of stretching on resize.
        pad_value: Border fill value used when letterboxing.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        normalization: NormalizationMode = NormalizationMode.RESCALE,
        margin: float = 0.0,
        input_is_bgr: bool = True,
        preserve_aspect_ratio: bool = False,
        pad_value: int = 0,
    ) -> None:
        """Initialise the preprocessor.

        Args:
            target_size: Model input as ``(height, width)``. Defaults to ``(256, 256)``.
            normalization: See :class:`NormalizationMode`. Defaults to ``RESCALE``.
            margin: Context margin as a fraction of box size. Defaults to ``0.0``.
            input_is_bgr: Set ``False`` if frames are already RGB. Defaults to ``True``.
            preserve_aspect_ratio: Letterbox instead of stretch. Defaults to ``False``.
            pad_value: Letterbox border value. Defaults to ``0``.

        Raises:
            ValueError: If ``margin`` is negative or ``target_size`` is invalid.
        """
        if margin < 0.0:
            raise ValueError(f"margin must be >= 0, got {margin!r}.")
        if len(target_size) != 2 or any(d <= 0 for d in target_size):
            raise ValueError(
                f"target_size must be two positive ints (h, w), got {target_size!r}."
            )

        self.target_size = target_size
        self.normalization = normalization
        self.margin = margin
        self.input_is_bgr = input_is_bgr
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.pad_value = pad_value

    def preprocess(self, frame: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
        """Run the full pipeline for one detected face.

        Args:
            frame: Raw camera frame, ``(H, W, 3)`` ``uint8`` (BGR unless configured
                otherwise).
            bbox: ``(x1, y1, x2, y2)`` absolute-pixel box from the detector.

        Returns:
            A ``(1, target_h, target_w, 3)`` ``float32`` array ready for the
            classifier.

        Raises:
            ValueError: If ``frame`` is not a 3-channel ``uint8`` ndarray, or the
                crop is empty after boundary clamping.
        """
        self._validate_frame(frame)
        face = crop_face(frame, bbox, margin=self.margin)
        return self.preprocess_crop(face)

    def preprocess_crop(self, face: np.ndarray) -> np.ndarray:
        """Run the pipeline on an already-cropped face (no detection/crop step).

        Useful for unit tests and for reusing preprocessing outside the detector
        flow (e.g. still-image inputs).

        Args:
            face: Cropped face image, ``(h, w, 3)`` ``uint8``.

        Returns:
            A ``(1, target_h, target_w, 3)`` ``float32`` array ready for inference.

        Raises:
            ValueError: If ``face`` is not a 3-channel ``uint8`` ndarray.
        """
        self._validate_frame(face)
        resized = resize_image(
            face,
            size=self.target_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            pad_value=self.pad_value,
        )
        rgb = bgr_to_rgb(resized) if self.input_is_bgr else resized
        normalized = normalize_image(rgb, self.normalization)
        return add_batch_dim(normalized)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate an input image is a 3-channel ``uint8`` ndarray.

        Args:
            frame: The candidate image.

        Raises:
            ValueError: If the type, rank, channel count, or dtype is wrong.
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"image must be a numpy.ndarray, got {type(frame).__name__}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {frame.shape!r}.")
        if frame.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got dtype {frame.dtype!r}.")
