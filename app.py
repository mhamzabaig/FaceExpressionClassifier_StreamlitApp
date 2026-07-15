"""Streamlit web app: real-time facial-emotion recognition over WebRTC.

This file is **UI only**. All inference lives behind
:class:`src.presentation.video_processor.EmotionVideoProcessor` (which wraps the
UI-agnostic :class:`src.application.WebcamPipeline`), so this module never
touches MediaPipe, TFLite, or OpenCV directly.

Run it with::

    streamlit run app.py

Design choices called out in the requirements:
    * **UI / inference separation** — see the module docstring above.
    * **session_state** — a single boolean ``camera_on`` is the source of truth
      for the camera; the Start/Stop buttons flip it via ``on_click`` callbacks
      and it drives ``webrtc_streamer(desired_playing_state=...)``.
    * **Avoid unnecessary reruns** — the live stats live in an
      ``st.fragment(run_every=...)`` that reruns *itself* a couple of times a
      second while the camera is on, instead of rerunning the whole script.
      When the camera is off, nothing polls.
"""

from __future__ import annotations

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.config import MODEL_NAME
from src.presentation import EmotionVideoProcessor

# ── Static UI content ────────────────────────────────────────────────────────

PAGE_TITLE = "Facial Emotion Recognition"
PAGE_ICON = "🎭"

PROJECT_DESCRIPTION = """
A real-time **facial-emotion recognition** web app. Your webcam stream stays in
your browser and is processed frame-by-frame over **WebRTC**: each frame is
scanned for faces, every face is cropped and classified into an emotion, and the
result is drawn back onto the video you see — all live, with no images stored or
uploaded anywhere.
"""

TECH_STACK = [
    ("Streamlit", "Web UI & app hosting"),
    ("streamlit-webrtc", "Low-latency webcam streaming in the browser"),
    ("MediaPipe (BlazeFace)", "Fast CPU multi-face detection"),
    ("TensorFlow Lite / LiteRT", "Emotion classifier inference"),
    ("OpenCV", "Crop / resize / annotation"),
    ("NumPy", "Array plumbing"),
]

#: Client-side rendered DOT graph — no system Graphviz needed.
ARCHITECTURE_DOT = """
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#eef2ff",
          color="#6366f1", fontname="Helvetica", fontsize=11, margin=0.15];
    edge [color="#94a3b8"];

    cam   [label="Webcam\\n(browser)", fillcolor="#dcfce7", color="#22c55e"];
    det   [label="Face Detection\\n(MediaPipe)"];
    prep  [label="Crop + Resize\\n+ Normalize"];
    clf   [label="Emotion Classifier\\n(TFLite / Keras)", fillcolor="#fee2e2", color="#ef4444"];
    draw  [label="Draw Boxes\\n& Labels"];
    out   [label="Annotated Frame\\n(your screen)", fillcolor="#dcfce7", color="#22c55e"];

    cam -> det -> prep -> clf -> draw -> out;
}
"""

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


# ── session_state ────────────────────────────────────────────────────────────

def _init_state() -> None:
    """Initialise session_state keys exactly once per session."""
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False


def _start_camera() -> None:
    """Start-button callback: request the stream to play."""
    st.session_state.camera_on = True


def _stop_camera() -> None:
    """Stop-button callback: request the stream to stop."""
    st.session_state.camera_on = False


# ── UI sections ──────────────────────────────────────────────────────────────

def render_landing() -> None:
    """Render the header, description, tech stack and architecture diagram."""
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown(PROJECT_DESCRIPTION)

    with st.expander("🧰 Technology stack & architecture", expanded=True):
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Technology stack")
            for name, purpose in TECH_STACK:
                st.markdown(f"- **{name}** — {purpose}")
        with right:
            st.subheader("Architecture")
            st.graphviz_chart(ARCHITECTURE_DOT, use_container_width=True)


def render_controls() -> None:
    """Render the Start/Stop camera buttons wired to session_state."""
    start_col, stop_col, status_col = st.columns([1, 1, 3])
    start_col.button(
        "▶ Start Camera",
        on_click=_start_camera,
        disabled=st.session_state.camera_on,
        use_container_width=True,
        type="primary",
    )
    stop_col.button(
        "⏹ Stop Camera",
        on_click=_stop_camera,
        disabled=not st.session_state.camera_on,
        use_container_width=True,
    )
    with status_col:
        if st.session_state.camera_on:
            st.success("Camera requested — allow browser access if prompted.")
        else:
            st.info("Camera is off.")


def _render_metrics(metrics) -> None:
    """Draw the FPS / face-count / per-face results panel.

    Args:
        metrics: A ``StreamMetrics`` snapshot, or ``None`` when the camera is off.
    """
    fps_col, faces_col = st.columns(2)
    if metrics is None:
        fps_col.metric("Live FPS", "—")
        faces_col.metric("Faces detected", "—")
        st.caption("Start the camera to see live inference results.")
        return

    fps_col.metric("Live FPS", f"{metrics.fps:.1f}")
    faces_col.metric("Faces detected", metrics.face_count)

    st.markdown("**Inference results**")
    if not metrics.results:
        st.caption("No faces in the current frame.")
        return

    for index, result in enumerate(metrics.results, start=1):
        if result["class_id"] < 0:
            st.write(f"Face {index}: _classification failed_")
            continue
        confidence = float(result["confidence"])
        st.write(f"Face {index}: **{result['label']}** — {confidence * 100:.0f}%")
        st.progress(min(max(confidence, 0.0), 1.0))


@st.fragment(run_every=0.5)
def render_live_stats(ctx) -> None:
    """Poll the processor for metrics and redraw — a fragment, not a full rerun.

    Runs every 0.5 s **only while mounted** (we mount it only when the camera is
    on), so the whole-app script does not rerun for the FPS counter.

    Args:
        ctx: The value returned by :func:`webrtc_streamer`.
    """
    metrics = None
    if ctx is not None and ctx.state.playing and ctx.video_processor is not None:
        metrics = ctx.video_processor.get_metrics()
    _render_metrics(metrics)


# ── Page ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Compose the page."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    _init_state()

    render_landing()
    st.divider()
    render_controls()

    feed_col, stats_col = st.columns([3, 2])

    with feed_col:
        st.subheader("Camera feed")
        ctx = webrtc_streamer(
            key="emotion-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            desired_playing_state=st.session_state.camera_on,
            async_processing=True,
        )

    with stats_col:
        st.subheader("Live results")
        if st.session_state.camera_on:
            render_live_stats(ctx)   # self-refreshing fragment
        else:
            _render_metrics(None)    # static idle panel — no polling

    st.caption(f"Model in use: `{MODEL_NAME}` · processing runs on the server, per frame.")


if __name__ == "__main__":
    main()
