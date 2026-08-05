"""Gradio interface for the SignLearn isolated-sign practice assistant."""

from __future__ import annotations

import html
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import gradio as gr
import gradio.components.video as gradio_video_component
from imageio_ffmpeg import get_ffmpeg_exe

from signlearn import SignLearnEngine

LOGGER = logging.getLogger("signlearn.app")
AnalysisOutput = tuple[str, dict[str, float], dict[str, Any]]


def _configure_gradio_ffmpeg() -> None:
    """Make Gradio use the FFmpeg executable bundled with imageio-ffmpeg."""
    ffmpeg_class = gradio_video_component.FFmpeg
    gradio_video_component.FFmpeg = partial(
        ffmpeg_class,
        executable=get_ffmpeg_exe(),
    )


_configure_gradio_ffmpeg()

ROOT = Path(__file__).resolve().parent
ENGINE = SignLearnEngine(ROOT / "models", ROOT / "assets")
VOCABULARY = sorted(ENGINE.labels)


def _video_filepath(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("video") or value.get("path")
    if isinstance(value, tuple | list) and value:
        return value[0]
    return str(value)


def _empty_outputs(message: str) -> AnalysisOutput:
    return (
        f"### Please retry\n\n{message}",
        {},
        {"quality_passed": False, "issues": [message]},
    )


def analyze_video(video: Any, practice_target: str | None, mirror: bool) -> AnalysisOutput:
    video_path = _video_filepath(video)
    if not video_path:
        return _empty_outputs("Record or upload a short video first.")

    try:
        result = ENGINE.analyze(video_path, mirror=mirror)
    except Exception:
        LOGGER.exception("Video analysis failed")
        return _empty_outputs(
            "This video could not be processed. Check the terminal for details, then try another "
            "recording."
        )

    quality = result["quality"]
    if not quality["quality_passed"]:
        issue_lines = "\n".join(f"- {html.escape(issue)}" for issue in quality["issues"])
        return (
            "### Recording quality needs attention\n\n"
            f"{issue_lines}\n\nNo sign prediction was made from this clip.",
            {},
            quality,
        )

    predictions = result["top_predictions"]
    scores = {item["label"]: item["confidence"] for item in predictions}
    best = predictions[0]
    target = practice_target if practice_target in ENGINE.labels else None

    if not result["accepted"]:
        message = (
            "### Not confident enough — please retry\n\n"
            f"The leading guess was **{best['label']}** at **{best['confidence']:.1%}**, "
            f"below the operational **{result['threshold']:.0%}** acceptance threshold. "
            "Try one clear, complete sign with your upper body centered."
        )
    elif target and best["label"] == target:
        message = (
            "### Target recognized ✓\n\n"
            f"The model recognized **{best['label']}** with **{best['confidence']:.1%}** confidence."
        )
    elif target:
        message = (
            "### Different sign recognized\n\n"
            f"Your target was **{target}**, while the model recognized **{best['label']}** "
            f"with **{best['confidence']:.1%}** confidence. Review the sign and try again."
        )
    else:
        message = (
            "### Sign recognized\n\n"
            f"The model recognized **{best['label']}** with **{best['confidence']:.1%}** confidence."
        )

    message += (
        "\n\n*This is model feedback for practice—not a correctness guarantee or a substitute "
        "for instruction from Deaf/ASL educators.*"
    )
    quality["neural_network_inference_ms"] = result.get("inference_ms")
    quality["total_processing_ms"] = result.get("total_processing_ms")
    quality["accepted_at_threshold"] = result["accepted"]
    quality["operational_confidence_threshold"] = result["threshold"]
    quality["validation_selected_threshold"] = result["validation_selected_threshold"]
    return message, scores, quality


CSS = """
.gradio-container { max-width: 1120px !important; }
.hero { padding: 1.35rem 1.5rem; border-radius: 18px; color: white;
  background: linear-gradient(120deg, #172554, #0f766e); margin-bottom: 1rem; }
.hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; }
.hero p { margin: 0; opacity: .92; }
.tip { border-left: 4px solid #14b8a6; padding-left: .9rem; }
"""


with gr.Blocks(title="SignLearn ASL Practice Assistant") as demo:
    gr.HTML(
        "<section class='hero'><h1>SignLearn</h1>"
        "<p>A quality-aware practice assistant for 50 isolated ASL signs.</p></section>"
    )
    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(
                label="Record or upload one isolated sign",
                sources=["webcam", "upload"],
                format="mp4",
                include_audio=False,
            )
            with gr.Row():
                target = gr.Dropdown(
                    choices=VOCABULARY,
                    value=None,
                    label="Practice target (optional)",
                    filterable=True,
                )
                mirror = gr.Checkbox(
                    value=False,
                    label="My camera recording is mirrored",
                    info="Choose once for this camera session; do not change it separately for each sign.",
                )
            analyze = gr.Button("Analyze my sign", variant="primary", size="lg")
        with gr.Column(scale=2):
            result_text = gr.Markdown(
                "### Ready\n\nChoose a target if you wish, then record one sign."
            )
            predictions = gr.Label(label="Top 3 model predictions", num_top_classes=3)

    with gr.Accordion("Recording guide", open=True):
        gr.Markdown(
            """
<div class="tip">

1. Record one complete sign for about **1–3 seconds**.
2. Keep your **face, both shoulders, torso, and signing hand(s)** in frame.
3. Use even front lighting and a simple background.
4. Start and finish in a neutral position; avoid extra movements.
5. Set the mirrored-camera option once and keep it unchanged throughout the session.

</div>
            """
        )

    with gr.Accordion("Tracking and quality details", open=False):
        quality = gr.JSON(label="Quality report")

    with gr.Accordion("Supported 50-sign vocabulary", open=False):
        gr.Markdown(", ".join(f"`{label}`" for label in VOCABULARY))

    gr.Markdown(
        "**Scope:** isolated-sign recognition only. SignLearn does not translate sentences, "
        "assess ASL fluency, or replace a Deaf instructor or interpreter."
    )

    analyze.click(
        fn=analyze_video,
        inputs=[video, target, mirror],
        outputs=[result_text, predictions, quality],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("SIGNLEARN_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("SIGNLEARN_PORT", "7860")),
        show_error=True,
        css=CSS,
    )
