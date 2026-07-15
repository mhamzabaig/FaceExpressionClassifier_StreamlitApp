"""Central configuration — the single place to pick which model runs.

Set :data:`MODEL_NAME` to the file you want to use. Drop the file into the
``models/`` folder at the project root and the rest of the app adapts itself:

    * ``*.keras`` / ``*.h5`` / ``*.hdf5``  -> the Keras backend is used.
    * ``*.tflite`` / ``*.tflite16`` / ``*.lite`` -> the TFLite (LiteRT) backend.

The backend is chosen from the *file extension* by
:func:`src.infrastructure.inference.create_classifier`; you do not select it
here. Just change :data:`MODEL_NAME`.

.. warning::
    A ``.keras`` / ``.h5`` model requires TensorFlow (or standalone Keras) to be
    installed. The pinned ``requirements.txt`` ships only ``ai-edge-litert`` for
    a small Streamlit Cloud image, so a Keras model works locally but will fail
    to load on Cloud until ``tensorflow`` is added to ``requirements.txt``.
"""

from __future__ import annotations

import os
from typing import Tuple

# ── The one knob you change ──────────────────────────────────────────────────
#: File name of the desired model, expected inside :data:`MODELS_DIR`.
#: Change this to switch models, e.g. ``"Model.keras"`` or ``"model.tflite16"``.
MODEL_NAME: str = "emotion_model.tflite"

#: Spatial input size the model expects, as ``(height, width)``.
MODEL_INPUT_SIZE: Tuple[int, int] = (256, 256)

# ── Paths ────────────────────────────────────────────────────────────────────
#: Project root: ``src/config.py`` -> up one level.
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Folder that holds the classifier model(s).
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")


def get_model_path(model_name: str = MODEL_NAME) -> str:
    """Resolve ``model_name`` to an absolute path on disk.

    Looks in :data:`MODELS_DIR` first, then falls back to the project root (so a
    model committed at the repo root still resolves without being moved).

    Args:
        model_name: File name of the model. Defaults to :data:`MODEL_NAME`.

    Returns:
        The absolute path to the model file.

    Raises:
        FileNotFoundError: If the file is in neither location. The message lists
            both paths that were tried.
    """
    in_models_dir = os.path.join(MODELS_DIR, model_name)
    if os.path.isfile(in_models_dir):
        return in_models_dir

    at_root = os.path.join(PROJECT_ROOT, model_name)
    if os.path.isfile(at_root):
        return at_root

    raise FileNotFoundError(
        f"Model {model_name!r} not found. Looked in:\n"
        f"  - {in_models_dir}\n"
        f"  - {at_root}\n"
        f"Place the file in the 'models/' folder and set MODEL_NAME in src/config.py."
    )
