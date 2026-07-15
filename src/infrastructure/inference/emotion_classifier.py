"""Backend-agnostic emotion classifier (TensorFlow Lite **or** Keras).

The public entry point is :func:`create_classifier`, which inspects the model
file's extension and returns the matching backend:

    * ``.tflite`` / ``.tflite16`` / ``.lite`` -> :class:`TFLiteEmotionClassifier`
      (LiteRT / tflite-runtime / ``tensorflow.lite``).
    * ``.keras`` / ``.h5`` / ``.hdf5``        -> :class:`KerasEmotionClassifier`
      (``tensorflow.keras`` or standalone ``keras``, imported lazily).

Both backends share one interface, so the caller never branches on model type::

    from src.config import get_model_path
    from src.infrastructure.inference import create_classifier

    with create_classifier(get_model_path()) as clf:
        result = clf.predict(preprocessed_tensor)  # {"class_id", "label", "confidence"}

Responsibilities (both backends):
    * Load the model once and prepare it for repeated inference.
    * Run inference on a preprocessed ``(1, H, W, 3)`` float32 tensor.
    * Return ``{"class_id": int, "label": str, "confidence": float}``.

Explicitly out of scope:
    * No preprocessing (see ``preprocessing.py``) — input must already be the
      model's expected shape and normalization.
    * No Streamlit, no OpenCV/drawing.

Thread-safety:
    Neither a TFLite interpreter nor a Keras call is guaranteed concurrency-safe,
    so each backend serializes inference behind a lock; :meth:`predict` is safe
    to call from multiple threads (calls are serialized, not parallelized).

.. note::
    A ``.keras`` model needs TensorFlow (or standalone Keras) installed. The
    pinned ``requirements.txt`` ships only ``ai-edge-litert``; add ``tensorflow``
    there before deploying a Keras model to Streamlit Cloud.
"""

from __future__ import annotations

import logging
import os
import threading
from types import TracebackType
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

#: Canonical label map (0=Angry, 1=Happy, 2=Sad). Centralised in config later.
DEFAULT_LABELS: Dict[int, str] = {0: "Angry", 1: "Happy", 2: "Sad"}

#: Extensions routed to the Keras backend.
KERAS_EXTENSIONS: Tuple[str, ...] = (".keras", ".h5", ".hdf5")

#: Extensions routed to the TFLite backend. ``.tflite16`` / ``.tflite32`` are
#: accepted as convenience aliases people use for fp16/fp32 models (both are
#: still a standard ``.tflite`` flatbuffer).
TFLITE_EXTENSIONS: Tuple[str, ...] = (".tflite", ".tflite16", ".tflite32", ".lite")


class EmotionClassifierError(RuntimeError):
    """Raised when the model cannot be loaded, or predict is used after close."""


class EmotionResult(TypedDict):
    """A single emotion prediction.

    Attributes:
        class_id: Index of the argmax class.
        label: Human-readable label for ``class_id``.
        confidence: Probability of the predicted class in ``[0.0, 1.0]``.
    """

    class_id: int
    label: str
    confidence: float


class _BaseEmotionClassifier:
    """Shared logic for all backends (postprocessing, labels, lifecycle).

    Subclasses implement the three backend-specific hooks:

        * :meth:`_prepare_input`  — cast/quantize the float tensor to what the
          backend's graph expects.
        * :meth:`_infer`          — run the graph and return the raw output array.
        * :meth:`_dequantize`     — map raw output back to float (identity unless
          the backend produces quantized output).

    Everything else — shape validation, softmax auto-detection, argmax, the lock,
    warmup, and the context-manager protocol — lives here so the two backends
    behave identically.

    Attributes:
        labels: Mapping from class id to label string.
        apply_softmax: Whether to softmax the raw output before reading confidence.
    """

    def __init__(
        self,
        labels: Optional[Mapping[int, str]],
        apply_softmax: Optional[bool],
    ) -> None:
        self.labels: Dict[int, str] = dict(labels) if labels is not None else dict(DEFAULT_LABELS)
        self.apply_softmax = apply_softmax
        self._lock = threading.Lock()
        self._closed = False
        # Concrete shape the model expects, set by the subclass before warmup.
        self._input_shape: Tuple[int, ...] = ()

    # ── Backend hooks (subclasses override) ──────────────────────────────────
    def _prepare_input(self, input_tensor: np.ndarray) -> np.ndarray:
        """Cast/quantize the validated float tensor to the backend's input dtype."""
        raise NotImplementedError

    def _infer(self, prepared: np.ndarray) -> np.ndarray:
        """Run one inference on ``prepared`` and return the raw output array."""
        raise NotImplementedError

    def _dequantize(self, raw_output: np.ndarray) -> np.ndarray:
        """Map raw output to float32. Identity unless the backend quantizes output."""
        # copy=False: no allocation when the output is already float32.
        return raw_output.astype(np.float32, copy=False)

    # ── Shared prediction path ───────────────────────────────────────────────
    def predict(self, input_tensor: np.ndarray) -> EmotionResult:
        """Predict the emotion for a single preprocessed face tensor.

        Args:
            input_tensor: A preprocessed array matching the model input shape
                (e.g. ``(1, 256, 256, 3)`` float32). Any dtype conversion the
                backend needs is handled internally.

        Returns:
            An :class:`EmotionResult` with ``class_id``, ``label``, ``confidence``.

        Raises:
            EmotionClassifierError: If called after :meth:`close`.
            ValueError: If ``input_tensor`` shape is incompatible with the model.
        """
        if self._closed:
            raise EmotionClassifierError("predict() called on a closed classifier.")

        self._validate_input_shape(input_tensor)
        prepared = self._prepare_input(input_tensor)

        # Serialize backend access: neither TFLite nor Keras is concurrency-safe.
        with self._lock:
            raw_output = self._infer(prepared)

        probs = self._postprocess(raw_output)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        label = self.labels.get(class_id, str(class_id))
        return EmotionResult(class_id=class_id, label=label, confidence=confidence)

    def _validate_input_shape(self, input_tensor: np.ndarray) -> None:
        """Ensure ``input_tensor`` matches the model's expected shape.

        Args:
            input_tensor: The array passed to :meth:`predict`.

        Raises:
            ValueError: If the shape does not match ``self._input_shape``.
        """
        if tuple(input_tensor.shape) != self._input_shape:
            raise ValueError(
                f"input_tensor shape {tuple(input_tensor.shape)} does not match "
                f"model input shape {self._input_shape}."
            )

    def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        """Dequantize (if needed) and convert raw output to a probability vector.

        Args:
            raw_output: The tensor read from the output node.

        Returns:
            A 1-D ``float32`` probability vector over classes.
        """
        logits = self._dequantize(raw_output).reshape(-1)
        if self.apply_softmax is True:
            return self._softmax(logits)
        if self.apply_softmax is False:
            return logits
        # Auto: if the output already looks like a probability distribution
        # (non-negative and sums to ~1), it has a softmax layer — don't re-apply.
        if logits.min() >= -1e-6 and abs(float(logits.sum()) - 1.0) < 1e-3:
            return logits
        return self._softmax(logits)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over a 1-D vector.

        Args:
            x: 1-D array of logits.

        Returns:
            A probability distribution summing to 1.
        """
        shifted = x - np.max(x)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def _warmup(self) -> None:
        """Run one dummy inference to prime the backend (first-call latency)."""
        try:
            dummy = np.zeros(self._input_shape, dtype=np.float32)
            self.predict(dummy)
            logger.info("%s warmup complete.", type(self).__name__)
        except Exception as exc:  # noqa: BLE001 - warmup must not be fatal
            logger.warning("%s warmup failed (non-fatal): %s", type(self).__name__, exc)

    def close(self) -> None:
        """Release backend resources. Safe to call multiple times."""
        with self._lock:
            self._closed = True
        self._release()
        logger.info("%s closed.", type(self).__name__)

    def _release(self) -> None:
        """Backend-specific resource release. Overridden by subclasses."""

    def __enter__(self) -> "_BaseEmotionClassifier":
        """Enter the runtime context and return the classifier."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the runtime context, releasing resources."""
        self.close()


def _create_interpreter(model_path: str, num_threads: int) -> Any:
    """Create a TFLite interpreter from the first available backend.

    Tries LiteRT, then tflite-runtime, then TensorFlow, so the same code path
    works on the Cloud deploy target and on a local dev machine.

    Args:
        model_path: Filesystem path to the ``.tflite`` model.
        num_threads: Number of CPU threads for inference.

    Returns:
        An allocated-but-not-yet-``allocate_tensors()`` interpreter instance.

    Raises:
        EmotionClassifierError: If no supported TFLite backend is installed.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        logger.info("Using ai-edge-litert (LiteRT) backend.")
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        logger.info("Using tflite-runtime backend.")
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore

        logger.info("Using tensorflow.lite backend.")
        return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError as exc:
        raise EmotionClassifierError(
            "No TFLite backend found. Install one of: ai-edge-litert, "
            "tflite-runtime, or tensorflow."
        ) from exc


class TFLiteEmotionClassifier(_BaseEmotionClassifier):
    """Thread-safe, reusable TFLite/LiteRT emotion classifier.

    The model is loaded and tensors allocated once in :meth:`__init__`; every
    :meth:`predict` reuses the same interpreter. Handles float and int8/uint8
    models by reading the interpreter's own quantization parameters.

    Attributes:
        num_threads: CPU threads used by the interpreter.
    """

    def __init__(
        self,
        model_path: str,
        labels: Optional[Mapping[int, str]] = None,
        input_size: Tuple[int, int] = (256, 256),
        num_threads: Optional[int] = None,
        apply_softmax: Optional[bool] = None,
        warmup: bool = True,
    ) -> None:
        """Load the model once and allocate tensors once.

        Args:
            model_path: Path to the ``.tflite`` model file.
            labels: Optional class-id -> label map. Defaults to
                :data:`DEFAULT_LABELS` (0=Angry, 1=Happy, 2=Sad).
            input_size: Spatial input size ``(height, width)`` the model expects.
                Dynamic-input models (``shape_signature`` with ``-1``) are resized
                to ``[1, height, width, channels]`` before allocation.
            num_threads: CPU threads for inference. Defaults to ``os.cpu_count()``.
            apply_softmax: Whether to softmax the raw output before reading
                confidence. ``None`` (default) auto-detects. ``class_id`` (argmax)
                is unaffected either way.
            warmup: If ``True`` (default), run one dummy inference so the first
                real prediction isn't penalised by lazy allocation.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            EmotionClassifierError: If no TFLite backend is available or the model
                fails to load/allocate.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path!r}")

        super().__init__(labels=labels, apply_softmax=apply_softmax)
        self.input_size = input_size
        self.num_threads = num_threads if num_threads is not None else (os.cpu_count() or 2)

        try:
            self._interpreter = _create_interpreter(model_path, self.num_threads)
            self._resize_input_if_needed(input_size)  # handle dynamic-shape models
            self._interpreter.allocate_tensors()  # allocate ONCE (after any resize)
        except EmotionClassifierError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any load failure
            raise EmotionClassifierError(
                f"Failed to load/allocate TFLite model {model_path!r}: {exc}"
            ) from exc

        self._input_detail = self._interpreter.get_input_details()[0]
        self._output_detail = self._interpreter.get_output_details()[0]
        self._input_index: int = self._input_detail["index"]
        self._output_index: int = self._output_detail["index"]
        self._input_dtype: np.dtype = self._input_detail["dtype"]
        self._output_dtype: np.dtype = self._output_detail["dtype"]
        self._input_shape = tuple(int(d) for d in self._input_detail["shape"])

        logger.info(
            "TFLiteEmotionClassifier loaded: input%s %s, output %s, threads=%d",
            self._input_shape,
            self._input_dtype,
            self._output_dtype,
            self.num_threads,
        )

        if warmup:
            self._warmup()

    def _resize_input_if_needed(self, input_size: Tuple[int, int]) -> None:
        """Resize the input tensor to ``[1, H, W, C]`` for dynamic-shape models.

        Args:
            input_size: Target ``(height, width)``.
        """
        detail = self._interpreter.get_input_details()[0]
        current = [int(d) for d in detail["shape"]]
        signature = [int(d) for d in detail["shape_signature"]]
        channels = signature[-1] if signature[-1] > 0 else (current[-1] if current[-1] > 0 else 3)
        desired = [1, input_size[0], input_size[1], channels]

        if current == desired:
            return

        logger.info("Resizing model input from %s to %s (dynamic-shape model).", current, desired)
        self._interpreter.resize_tensor_input(detail["index"], desired, strict=False)

    def _prepare_input(self, input_tensor: np.ndarray) -> np.ndarray:
        """Quantize/cast the input to the interpreter's expected dtype.

        Args:
            input_tensor: Preprocessed float array (already shape-validated).

        Returns:
            An array in the interpreter's expected dtype.
        """
        scale, zero_point = self._input_detail.get("quantization", (0.0, 0))
        if np.issubdtype(self._input_dtype, np.integer) and scale not in (0, 0.0):
            # Real -> quantized: q = round(x / scale + zero_point), clipped to dtype range.
            info = np.iinfo(self._input_dtype)
            quantized = np.round(input_tensor / scale + zero_point)
            quantized = np.clip(quantized, info.min, info.max)
            return quantized.astype(self._input_dtype)
        # copy=False: the preprocessed tensor is already float32, so this is a
        # no-op for a float model (set_tensor copies into the interpreter anyway).
        return input_tensor.astype(self._input_dtype, copy=False)

    def _infer(self, prepared: np.ndarray) -> np.ndarray:
        """Run the interpreter. Copies the output (get_tensor returns a view)."""
        self._interpreter.set_tensor(self._input_index, prepared)
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_index).copy()

    def _dequantize(self, raw_output: np.ndarray) -> np.ndarray:
        """Dequantize integer output using the interpreter's parameters."""
        scale, zero_point = self._output_detail.get("quantization", (0.0, 0))
        output = raw_output.astype(np.float32, copy=False)
        if np.issubdtype(self._output_dtype, np.integer) and scale not in (0, 0.0):
            output = (output - zero_point) * scale
        return output

    def _release(self) -> None:
        """Drop the interpreter reference."""
        self._interpreter = None


class KerasEmotionClassifier(_BaseEmotionClassifier):
    """Thread-safe, reusable Keras emotion classifier.

    Loads a ``.keras`` / ``.h5`` model once and runs eager inference per call.
    TensorFlow (or standalone Keras) is imported lazily, so importing this module
    never pulls in TensorFlow unless a Keras model is actually used.

    Attributes:
        input_size: Spatial input size ``(height, width)`` the model expects.
    """

    def __init__(
        self,
        model_path: str,
        labels: Optional[Mapping[int, str]] = None,
        input_size: Tuple[int, int] = (256, 256),
        num_threads: Optional[int] = None,  # accepted for API parity; unused
        apply_softmax: Optional[bool] = None,
        warmup: bool = True,
    ) -> None:
        """Load the Keras model once.

        Args:
            model_path: Path to the ``.keras`` / ``.h5`` model file.
            labels: Optional class-id -> label map. Defaults to :data:`DEFAULT_LABELS`.
            input_size: Spatial input size ``(height, width)``. Used to build the
                expected input shape when the model reports an unknown (``None``)
                spatial dimension. Defaults to ``(256, 256)``.
            num_threads: Ignored (kept so the constructor signature matches the
                TFLite backend and the factory can forward kwargs uniformly).
            apply_softmax: Whether to softmax the raw output. ``None`` auto-detects.
            warmup: If ``True`` (default), run one dummy inference to prime graph
                tracing so the first real prediction isn't penalised.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            EmotionClassifierError: If TensorFlow/Keras is not installed or the
                model fails to load.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path!r}")

        super().__init__(labels=labels, apply_softmax=apply_softmax)
        self.input_size = input_size

        load_model = self._resolve_loader()
        try:
            self._model = load_model(model_path, compile=False)
        except Exception as exc:  # noqa: BLE001 - normalise any load failure
            raise EmotionClassifierError(
                f"Failed to load Keras model {model_path!r}: {exc}"
            ) from exc

        self._input_shape = self._infer_input_shape(input_size)
        # Keras eager output is float32; base _dequantize (identity) is correct.
        self._input_dtype = np.float32

        logger.info(
            "KerasEmotionClassifier loaded: input%s float32, model=%s",
            self._input_shape,
            os.path.basename(model_path),
        )

        if warmup:
            self._warmup()

    @staticmethod
    def _resolve_loader() -> Any:
        """Return a ``load_model`` callable from TensorFlow or standalone Keras.

        Returns:
            A callable ``load_model(path, compile=...)``.

        Raises:
            EmotionClassifierError: If neither TensorFlow nor Keras is installed.
        """
        try:
            import tensorflow as tf  # type: ignore

            logger.info("Using tensorflow.keras backend for Keras model.")
            return tf.keras.models.load_model
        except ImportError:
            pass

        try:
            import keras  # type: ignore

            logger.info("Using standalone keras backend for Keras model.")
            return keras.saving.load_model
        except ImportError as exc:
            raise EmotionClassifierError(
                "Loading a .keras/.h5 model requires TensorFlow or Keras. Install "
                "one (e.g. `pip install tensorflow`) and add it to requirements.txt "
                "before deploying. The TFLite path does not need this."
            ) from exc

    def _infer_input_shape(self, input_size: Tuple[int, int]) -> Tuple[int, ...]:
        """Derive the concrete ``(1, H, W, C)`` input shape from the model.

        Falls back to ``input_size`` for any dimension the model leaves as
        ``None`` (dynamic).

        Args:
            input_size: Fallback ``(height, width)``.

        Returns:
            The concrete batch-1 input shape.
        """
        try:
            shape = tuple(self._model.input_shape)  # e.g. (None, 256, 256, 3)
        except (AttributeError, TypeError):
            shape = (None, input_size[0], input_size[1], 3)

        height = shape[1] if len(shape) > 1 and shape[1] else input_size[0]
        width = shape[2] if len(shape) > 2 and shape[2] else input_size[1]
        channels = shape[3] if len(shape) > 3 and shape[3] else 3
        return (1, int(height), int(width), int(channels))

    def _prepare_input(self, input_tensor: np.ndarray) -> np.ndarray:
        """Cast to float32 (Keras models take float input; no quantization)."""
        # copy=False: no-op when the preprocessed tensor is already float32.
        return input_tensor.astype(np.float32, copy=False)

    def _infer(self, prepared: np.ndarray) -> np.ndarray:
        """Run eager inference and return the output as a NumPy array."""
        # Call the model directly (eager) — faster than .predict() for a single
        # sample and avoids .predict()'s batching/logging overhead.
        output = self._model(prepared, training=False)
        return np.asarray(output)

    def _release(self) -> None:
        """Drop the model reference."""
        self._model = None


def create_classifier(model_path: str, **kwargs: Any) -> _BaseEmotionClassifier:
    """Build the right classifier backend for ``model_path`` (chosen by extension).

    Dispatch:
        * :data:`KERAS_EXTENSIONS`  -> :class:`KerasEmotionClassifier`
        * :data:`TFLITE_EXTENSIONS` -> :class:`TFLiteEmotionClassifier`

    Args:
        model_path: Path to the model file. The extension decides the backend.
        **kwargs: Forwarded to the chosen backend's constructor (``labels``,
            ``input_size``, ``num_threads``, ``apply_softmax``, ``warmup``).

    Returns:
        A ready-to-use classifier exposing :meth:`predict`, :meth:`close`, and the
        context-manager protocol.

    Raises:
        EmotionClassifierError: If the extension is not recognised.
        FileNotFoundError: If the file does not exist.
    """
    ext = os.path.splitext(model_path)[1].lower()

    if ext in KERAS_EXTENSIONS:
        logger.info("Dispatching %s to KerasEmotionClassifier.", os.path.basename(model_path))
        return KerasEmotionClassifier(model_path, **kwargs)

    if ext in TFLITE_EXTENSIONS:
        logger.info("Dispatching %s to TFLiteEmotionClassifier.", os.path.basename(model_path))
        return TFLiteEmotionClassifier(model_path, **kwargs)

    raise EmotionClassifierError(
        f"Unsupported model extension {ext!r} for {model_path!r}. "
        f"Use one of: {', '.join(KERAS_EXTENSIONS + TFLITE_EXTENSIONS)}."
    )
