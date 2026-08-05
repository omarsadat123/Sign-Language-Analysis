"""End-to-end SignLearn inference engine."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import preprocess_landmark_frames
from .video import extract_landmarks


class SignLearnEngine:
    """Lazy-load the trained network and run quality-aware inference."""

    def __init__(self, model_directory: Path, asset_directory: Path):
        self.model_directory = Path(model_directory)
        self.asset_directory = Path(asset_directory)
        self.manifest = self._read_json("deployment_manifest.json")
        self._verify_model_checksum()
        self.configuration = self._read_json("temporal_configuration.json")
        feature_configuration_path = self.model_directory / "feature_configuration.json"
        self.feature_configuration = (
            self._read_json("feature_configuration.json")
            if feature_configuration_path.exists()
            else None
        )
        self.expected_input_shape = tuple(int(value) for value in self.manifest["input_shape"][1:])
        raw_labels = self._read_json("label_map.json")
        self.labels = [name for name, _ in sorted(raw_labels.items(), key=lambda item: item[1])]
        self.validation_threshold = float(self.manifest["confidence_threshold"])
        self.threshold = float(
            self.manifest.get("operational_confidence_threshold", self.validation_threshold)
        )
        self._model = None
        self._model_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _read_json(self, filename: str) -> Any:
        with (self.model_directory / filename).open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _verify_model_checksum(self) -> None:
        expected = self.manifest.get("model_sha256")
        if not expected:
            return

        digest = hashlib.sha256()
        model_path = self.model_directory / "signlearn_model.keras"
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != str(expected).lower():
            raise RuntimeError(
                "The model checksum does not match the deployment manifest. "
                "Download a clean copy of the repository before running inference."
            )

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model

            import tensorflow as tf

            self._model = tf.keras.models.load_model(
                self.model_directory / "signlearn_model.keras", compile=False
            )
        return self._model

    def analyze(self, video_path: str, mirror: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        landmark_frames, duration = extract_landmarks(
            video_path,
            self.asset_directory / "holistic_landmarker.task",
            mirror=mirror,
        )
        features, quality = preprocess_landmark_frames(
            landmark_frames,
            self.configuration,
            duration_seconds=duration,
            feature_configuration=self.feature_configuration,
        )
        if tuple(features.shape[1:]) != self.expected_input_shape:
            raise ValueError(
                f"Preprocessing produced {features.shape}; model expects "
                f"(1, {', '.join(str(value) for value in self.expected_input_shape)})."
            )

        result: dict[str, Any] = {
            "quality": quality,
            "threshold": self.threshold,
            "validation_selected_threshold": self.validation_threshold,
            "accepted": False,
            "top_predictions": [],
        }
        if not quality["quality_passed"]:
            result["total_processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return result

        model = self._load_model()
        inference_started = time.perf_counter()
        with self._predict_lock:
            probabilities = np.asarray(model(features, training=False))[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000

        top_indices = np.argsort(probabilities)[::-1][:3]
        result["top_predictions"] = [
            {"label": self.labels[index], "confidence": float(probabilities[index])}
            for index in top_indices
        ]
        result["accepted"] = bool(probabilities[top_indices[0]] >= self.threshold)
        result["inference_ms"] = round(inference_ms, 1)
        result["total_processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result
