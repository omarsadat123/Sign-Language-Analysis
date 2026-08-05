"""Video decoding and MediaPipe Holistic extraction."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import LandmarkFrame, frame_from_mediapipe

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MAX_TRACKED_FRAMES = 240


def ensure_holistic_model(asset_path: Path) -> Path:
    """Download the official MediaPipe task model once, using an atomic rename."""
    if asset_path.exists() and asset_path.stat().st_size > 1_000_000:
        return asset_path

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="holistic_", suffix=".task", dir=asset_path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, temporary_path)
        if temporary_path.stat().st_size <= 1_000_000:
            raise RuntimeError("The downloaded MediaPipe model file is incomplete.")
        os.replace(temporary_path, asset_path)
    except Exception as exc:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not obtain the MediaPipe Holistic model. Connect to the internet for the "
            "first run, or download holistic_landmarker.task into the app's assets folder."
        ) from exc
    return asset_path


def _decode_video(
    video_path: str, mirror: bool
) -> tuple[list[tuple[int, np.ndarray]], float, float]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(
            "The video could not be opened. Try recording it again or upload MP4/WebM."
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    frames: list[tuple[int, np.ndarray]] = []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if mirror:
                rgb = cv2.flip(rgb, 1)
            frames.append((index, np.ascontiguousarray(rgb)))
            index += 1
            if index >= int(fps * 12):
                break
    finally:
        capture.release()

    if not frames:
        raise ValueError("No frames could be decoded from this recording.")

    duration_seconds = len(frames) / fps
    if len(frames) > MAX_TRACKED_FRAMES:
        chosen = np.rint(np.linspace(0, len(frames) - 1, MAX_TRACKED_FRAMES)).astype(int)
        frames = [frames[position] for position in chosen]
    return frames, duration_seconds, fps


def extract_landmarks(
    video_path: str,
    model_asset_path: Path,
    mirror: bool = False,
) -> tuple[list[LandmarkFrame], float]:
    """Track holistic landmarks for an uploaded or browser-recorded video."""
    import mediapipe as mp

    model_path = ensure_holistic_model(model_asset_path)
    decoded, duration_seconds, fps = _decode_video(video_path, mirror)

    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    landmark_frames: list[LandmarkFrame] = []
    previous_timestamp = -1
    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for original_index, rgb in decoded:
            timestamp_ms = max(previous_timestamp + 1, int(round(original_index * 1000 / fps)))
            previous_timestamp = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)
            landmark_frames.append(frame_from_mediapipe(result))
    return landmark_frames, duration_seconds
