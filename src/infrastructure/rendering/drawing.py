"""OpenCV annotation for detection + emotion results (UI-framework agnostic).

Pure drawing on NumPy **BGR** frames using ``cv2`` primitives. This module knows
nothing about Streamlit, WebRTC, or the pipeline — you hand it a frame plus a
``(bbox, label, confidence)`` and it annotates the frame *in place* (and also
returns it, for convenience/chaining).

Responsibilities:
    * Draw bounding boxes.
    * Draw the emotion label and its confidence.
    * Pick a distinct colour per emotion.
    * Place the label box automatically so it never spills off-frame, flipping
      it inside the box near the top edge.
    * Choose black/white text automatically for contrast against the box colour.

Everything is configured once on :class:`EmotionDrawer` so the per-face hot path
is a couple of ``cv2`` calls with no per-frame allocation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

#: Per-emotion box colours in **BGR** (OpenCV order). Keys are matched
#: case-insensitively against the predicted label.
DEFAULT_EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "angry": (66, 66, 244),    # red
    "happy": (80, 200, 80),    # green
    "sad": (230, 150, 70),     # blue
}

#: Fallback colour for unknown labels (e.g. ``"?"`` when inference failed).
DEFAULT_COLOR: Tuple[int, int, int] = (180, 180, 180)  # grey

#: BGR colours used by :meth:`EmotionDrawer._text_color_for` for auto contrast.
_BLACK: Tuple[int, int, int] = (0, 0, 0)
_WHITE: Tuple[int, int, int] = (255, 255, 255)


class EmotionDrawer:
    """Reusable, configured annotator for faces + emotion predictions.

    Stateless between calls (no per-frame mutable state), so a single instance
    is safe to reuse for the whole stream.

    Attributes:
        colors: Case-insensitive label -> BGR colour map.
        default_color: Colour used when a label is not in ``colors``.
        box_thickness: Bounding-box line thickness in pixels.
        font: OpenCV ``HERSHEY`` font constant.
        font_scale: Text scale passed to ``cv2.putText``.
        font_thickness: Text stroke thickness.
        text_pad: Padding (px) inside the filled label box.
        show_confidence: Whether to append the confidence to the label.
        auto_text_color: Pick black/white text by box-colour luminance.
        text_color: Text colour used when ``auto_text_color`` is ``False``.
    """

    def __init__(
        self,
        colors: Optional[Mapping[str, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = DEFAULT_COLOR,
        box_thickness: int = 2,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        text_pad: int = 4,
        show_confidence: bool = True,
        auto_text_color: bool = True,
        text_color: Tuple[int, int, int] = _WHITE,
    ) -> None:
        """Initialise the drawer with a fixed style.

        Args:
            colors: Optional label -> BGR map. Keys are lower-cased for
                case-insensitive lookup. Defaults to :data:`DEFAULT_EMOTION_COLORS`.
            default_color: BGR colour for labels not present in ``colors``.
            box_thickness: Bounding-box thickness in px. Defaults to ``2``.
            font: OpenCV font constant. Defaults to ``FONT_HERSHEY_SIMPLEX``.
            font_scale: Font scale. Defaults to ``0.6``.
            font_thickness: Text stroke thickness. Defaults to ``1``.
            text_pad: Inner padding of the label box in px. Defaults to ``4``.
            show_confidence: Append ``" 0.97"``-style confidence. Defaults to ``True``.
            auto_text_color: Choose black/white text for contrast. Defaults to ``True``.
            text_color: Fixed text colour when ``auto_text_color`` is ``False``.
        """
        source = colors if colors is not None else DEFAULT_EMOTION_COLORS
        self.colors: Dict[str, Tuple[int, int, int]] = {k.lower(): tuple(v) for k, v in source.items()}
        self.default_color = tuple(default_color)
        self.box_thickness = box_thickness
        self.font = font
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_pad = text_pad
        self.show_confidence = show_confidence
        self.auto_text_color = auto_text_color
        self.text_color = tuple(text_color)

    def color_for(self, label: str) -> Tuple[int, int, int]:
        """Return the BGR colour for ``label`` (case-insensitive).

        Args:
            label: Emotion label, e.g. ``"Happy"``.

        Returns:
            The configured BGR colour, or :attr:`default_color` if unknown.
        """
        return self.colors.get(label.lower(), self.default_color)

    def draw(
        self,
        frame: np.ndarray,
        bbox: Sequence[int],
        label: str,
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """Annotate one face: bounding box + label/confidence bar.

        Drawn in place; the same ``frame`` is returned for chaining.

        Args:
            frame: Target ``(H, W, 3)`` ``uint8`` BGR image (written in place).
            bbox: ``(x1, y1, x2, y2)`` absolute-pixel box.
            label: Emotion label to display.
            confidence: Optional confidence in ``[0, 1]``; appended when
                :attr:`show_confidence` is set. Pass ``None`` to show the label
                only (e.g. when inference failed).

        Returns:
            The same ``frame``, annotated.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = self.color_for(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        caption = self._caption(label, confidence)
        self._draw_label(frame, caption, x1, y1, color, width, height)
        return frame

    def draw_many(
        self,
        frame: np.ndarray,
        items: Iterable[Mapping[str, object]],
    ) -> np.ndarray:
        """Annotate several faces in one call.

        Args:
            frame: Target BGR image (written in place).
            items: Iterable of mappings with ``"bbox"``, ``"label"`` and an
                optional ``"confidence"`` key (e.g. the pipeline's results).

        Returns:
            The same ``frame``, annotated.
        """
        for item in items:
            self.draw(
                frame,
                item["bbox"],  # type: ignore[arg-type]
                str(item["label"]),
                item.get("confidence"),  # type: ignore[arg-type]
            )
        return frame

    def _caption(self, label: str, confidence: Optional[float]) -> str:
        """Build the caption string for a face.

        Args:
            label: Emotion label.
            confidence: Optional confidence in ``[0, 1]``.

        Returns:
            ``"Happy 0.97"`` when a confidence is shown, else just the label.
        """
        if self.show_confidence and confidence is not None:
            return f"{label} {confidence:.2f}"
        return label

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x1: int,
        y1: int,
        color: Tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        """Draw a filled caption bar with auto-adjusted position.

        The bar sits just above the box's top-left corner. If that would clip
        off the top of the frame, it flips to just *inside* the box top instead;
        it is also shifted left to stay within the frame width and clamped
        vertically so it never leaves the frame.

        Args:
            frame: Target image (written in place).
            text: Caption to render.
            x1: Box left in px.
            y1: Box top in px.
            color: Bar fill colour (BGR).
            width: Frame width in px.
            height: Frame height in px.
        """
        pad = self.text_pad
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        bar_w = text_w + 2 * pad
        bar_h = text_h + baseline + 2 * pad

        # Horizontal: align with the box left, shifting left so it stays on-frame.
        bx1 = max(0, min(x1, width - bar_w)) if bar_w <= width else 0
        bx2 = min(bx1 + bar_w, width)

        # Vertical: prefer above the box; flip inside if it would clip the top.
        by1 = y1 - bar_h
        if by1 < 0:
            by1 = y1
        by1 = max(0, min(by1, height - bar_h)) if bar_h <= height else 0
        by2 = by1 + bar_h

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, cv2.FILLED)

        text_color = self._text_color_for(color) if self.auto_text_color else self.text_color
        text_org = (bx1 + pad, by1 + pad + text_h)
        cv2.putText(
            frame,
            text,
            text_org,
            self.font,
            self.font_scale,
            text_color,
            self.font_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _text_color_for(bg_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Pick black or white text for contrast against ``bg_bgr``.

        Uses the Rec. 601 luma of the background colour.

        Args:
            bg_bgr: Background (bar) colour in BGR.

        Returns:
            Black on light backgrounds, white on dark ones.
        """
        blue, green, red = bg_bgr
        luma = 0.114 * blue + 0.587 * green + 0.299 * red
        return _BLACK if luma > 140 else _WHITE
