"""Landmark preprocessing kept aligned with the Kaggle training notebook."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

BASE_FEATURE_COUNT = 189
MOTION_COORDINATE_COUNT = 144
ENHANCED_FEATURE_COUNT = 492


@dataclass(frozen=True)
class LandmarkFrame:
    """One frame of unnormalized MediaPipe landmarks."""

    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    face: np.ndarray


def _blank(count: int) -> np.ndarray:
    return np.full((count, 3), np.nan, dtype=np.float32)


def _landmark_array(value: Any, count: int) -> np.ndarray:
    """Convert either a flat or one-element nested MediaPipe list to an array."""
    if value is None or len(value) == 0:
        return _blank(count)

    landmarks = value
    if not hasattr(value[0], "x"):
        landmarks = value[0] if len(value[0]) else []

    result = _blank(count)
    for index, landmark in enumerate(landmarks[:count]):
        result[index] = (landmark.x, landmark.y, landmark.z)
    return result


def frame_from_mediapipe(result: Any) -> LandmarkFrame:
    """Create a stable frame representation from a HolisticLandmarker result."""
    return LandmarkFrame(
        left_hand=_landmark_array(result.left_hand_landmarks, 21),
        right_hand=_landmark_array(result.right_hand_landmarks, 21),
        pose=_landmark_array(result.pose_landmarks, 33),
        face=_landmark_array(result.face_landmarks, 478),
    )


def _selected_values(
    frames: Iterable[LandmarkFrame],
    hand_indices: list[int],
    pose_indices: list[int],
    lip_indices: list[int],
) -> np.ndarray:
    selected = []
    for frame in frames:
        selected.append(
            np.concatenate(
                [
                    frame.left_hand[hand_indices],
                    frame.right_hand[hand_indices],
                    frame.pose[pose_indices],
                    frame.face[lip_indices],
                ],
                axis=0,
            )
        )
    if not selected:
        raise ValueError("No landmark frames were produced from the video.")
    return np.asarray(selected, dtype=np.float32)


def _quality_report(
    presence: np.ndarray,
    pose_start: int,
    lip_start: int,
    left_shoulder_offset: int,
    right_shoulder_offset: int,
    frame_count: int,
    duration_seconds: float,
) -> dict[str, Any]:
    left_frame = presence[:, :21].mean(axis=1)
    right_frame = presence[:, 21:42].mean(axis=1)
    pose_frame = presence[:, pose_start:lip_start].mean(axis=1)
    lip_frame = presence[:, lip_start:].mean(axis=1)
    shoulder_frame = presence[:, left_shoulder_offset] & presence[:, right_shoulder_offset]
    any_hand_frame = (left_frame >= 0.5) | (right_frame >= 0.5)

    metrics = {
        "decoded_frames": int(frame_count),
        "duration_seconds": round(float(duration_seconds), 2),
        "left_hand_frame_ratio": round(float((left_frame >= 0.5).mean()), 3),
        "right_hand_frame_ratio": round(float((right_frame >= 0.5).mean()), 3),
        "any_hand_frame_ratio": round(float(any_hand_frame.mean()), 3),
        "pose_detection_ratio": round(float(pose_frame.mean()), 3),
        "lip_detection_ratio": round(float(lip_frame.mean()), 3),
        "shoulder_frame_ratio": round(float(shoulder_frame.mean()), 3),
    }

    issues: list[str] = []
    if frame_count < 6 or duration_seconds < 0.25:
        issues.append("Record a longer clip (about 1–3 seconds).")
    if duration_seconds > 8.0:
        issues.append("Use a shorter clip containing only one sign (maximum 8 seconds).")
    if metrics["any_hand_frame_ratio"] < 0.20:
        issues.append("Keep at least one complete hand visible for more of the clip.")
    if metrics["shoulder_frame_ratio"] < 0.50 or metrics["pose_detection_ratio"] < 0.50:
        issues.append("Move back so your face, torso, and both shoulders are visible.")
    if metrics["lip_detection_ratio"] < 0.50:
        issues.append("Keep your face visible and use even front lighting.")

    metrics["quality_passed"] = not issues
    metrics["issues"] = issues
    return metrics


def _vector_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(first - second), axis=-1) + np.float32(1e-8))


def build_motion_geometry_features(base_features: np.ndarray) -> np.ndarray:
    """Append the frozen SignLearn-08B motion/geometry contract to 189 base features."""
    features = np.asarray(base_features, dtype=np.float32)
    if features.shape != (48, BASE_FEATURE_COUNT):
        raise ValueError(
            f"Expected base features with shape (48, {BASE_FEATURE_COUNT}), got {features.shape}."
        )

    # Keep these offsets explicit: they are part of the serialized model's feature contract.
    # 0:126 = two hands, 126:144 = selected pose, 144:184 = lips, 184:189 = visibility.
    motion_coordinates = features[:, :MOTION_COORDINATE_COUNT]
    detection = features[:, 184:189]

    previous_coordinates = np.concatenate([motion_coordinates[:1], motion_coordinates[:-1]], axis=0)
    velocity = motion_coordinates - previous_coordinates
    previous_detection = np.concatenate([detection[:1], detection[:-1]], axis=0)
    velocity_masks = np.concatenate(
        [
            np.repeat(
                (detection[:, 0:1] > 0.5) & (previous_detection[:, 0:1] > 0.5),
                63,
                axis=1,
            ),
            np.repeat(
                (detection[:, 1:2] > 0.5) & (previous_detection[:, 1:2] > 0.5),
                63,
                axis=1,
            ),
            np.repeat(
                (detection[:, 2:3] > 0.5) & (previous_detection[:, 2:3] > 0.5),
                18,
                axis=1,
            ),
        ],
        axis=1,
    )
    velocity = np.where(velocity_masks, velocity, np.float32(0.0))

    previous_velocity = np.concatenate([velocity[:1], velocity[:-1]], axis=0)
    acceleration = velocity - previous_velocity
    previous_velocity_mask = np.concatenate([velocity_masks[:1], velocity_masks[:-1]], axis=0)
    acceleration = np.where(velocity_masks & previous_velocity_mask, acceleration, np.float32(0.0))
    velocity = np.clip(velocity, -5.0, 5.0)
    acceleration = np.clip(acceleration, -5.0, 5.0)

    left_hand = features[:, 0:63].reshape(-1, 21, 3)
    right_hand = features[:, 63:126].reshape(-1, 21, 3)
    pose = features[:, 126:144].reshape(-1, 9, 2)
    lips = features[:, 144:184].reshape(-1, 20, 2)
    left_wrist, right_wrist = left_hand[:, 0, :2], right_hand[:, 0, :2]
    nose, left_shoulder, right_shoulder = pose[:, 0], pose[:, 1], pose[:, 2]
    left_pose_wrist, right_pose_wrist = pose[:, 5], pose[:, 6]
    mouth = lips.mean(axis=1)
    fingertip_indices = np.asarray([4, 8, 12, 16, 20], dtype=np.int32)
    left_tips = left_hand[:, fingertip_indices, :2]
    right_tips = right_hand[:, fingertip_indices, :2]
    left_spread = _vector_distance(left_tips, left_wrist[:, None, :]).mean(axis=1)
    right_spread = _vector_distance(right_tips, right_wrist[:, None, :]).mean(axis=1)
    left_tip_to_mouth = _vector_distance(left_tips, mouth[:, None, :]).min(axis=1)
    right_tip_to_mouth = _vector_distance(right_tips, mouth[:, None, :]).min(axis=1)

    geometry = np.stack(
        [
            _vector_distance(left_wrist, right_wrist),
            _vector_distance(left_wrist, mouth),
            _vector_distance(right_wrist, mouth),
            _vector_distance(left_wrist, nose),
            _vector_distance(right_wrist, nose),
            _vector_distance(left_wrist, left_shoulder),
            _vector_distance(right_wrist, right_shoulder),
            left_spread,
            right_spread,
            _vector_distance(left_hand[:, 4, :2], left_hand[:, 8, :2]),
            _vector_distance(right_hand[:, 4, :2], right_hand[:, 8, :2]),
            _vector_distance(left_wrist, left_pose_wrist),
            _vector_distance(right_wrist, right_pose_wrist),
            left_tip_to_mouth,
            right_tip_to_mouth,
        ],
        axis=1,
    )

    left_valid = detection[:, 0] > 0.5
    right_valid = detection[:, 1] > 0.5
    pose_valid = detection[:, 2] > 0.5
    lip_valid = detection[:, 3] > 0.5
    geometry_masks = np.stack(
        [
            left_valid & right_valid,
            left_valid & lip_valid,
            right_valid & lip_valid,
            left_valid & pose_valid,
            right_valid & pose_valid,
            left_valid & pose_valid,
            right_valid & pose_valid,
            left_valid,
            right_valid,
            left_valid,
            right_valid,
            left_valid & pose_valid,
            right_valid & pose_valid,
            left_valid & lip_valid,
            right_valid & lip_valid,
        ],
        axis=1,
    )
    geometry = np.where(geometry_masks, np.clip(geometry, 0.0, 10.0), np.float32(0.0))
    enhanced = np.concatenate([features, velocity, acceleration, geometry], axis=1)
    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=10.0, neginf=-5.0).astype(np.float32)
    if enhanced.shape != (48, ENHANCED_FEATURE_COUNT):
        raise AssertionError(f"Enhanced feature contract produced {enhanced.shape}.")
    return enhanced


def preprocess_landmark_frames(
    frames: list[LandmarkFrame],
    configuration: dict[str, Any],
    duration_seconds: float,
    feature_configuration: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize, sample, and optionally enhance one landmark sequence."""
    hand_indices = configuration["hand_indices"]
    pose_indices = configuration["pose_indices"]
    lip_indices = configuration["lip_indices"]
    target_frames = int(configuration["frame_count"])
    clip_min, clip_max = configuration["coordinate_clip"]

    values = _selected_values(frames, hand_indices, pose_indices, lip_indices)
    presence = np.isfinite(values).all(axis=2)

    pose_start = len(hand_indices) * 2
    lip_start = pose_start + len(pose_indices)
    left_shoulder_offset = pose_start + pose_indices.index(11)
    right_shoulder_offset = pose_start + pose_indices.index(12)

    quality = _quality_report(
        presence,
        pose_start,
        lip_start,
        left_shoulder_offset,
        right_shoulder_offset,
        len(frames),
        duration_seconds,
    )

    left_shoulder = values[:, left_shoulder_offset, :]
    right_shoulder = values[:, right_shoulder_offset, :]
    shoulder_valid = (
        presence[:, left_shoulder_offset]
        & presence[:, right_shoulder_offset]
        & (np.linalg.norm(left_shoulder[:, :2] - right_shoulder[:, :2], axis=1) > 1e-6)
    )

    centers = (left_shoulder + right_shoulder) / 2.0
    scales = np.linalg.norm(left_shoulder[:, :2] - right_shoulder[:, :2], axis=1)
    if shoulder_valid.any():
        fallback_center = np.nanmedian(centers[shoulder_valid], axis=0)
        fallback_scale = float(np.nanmedian(scales[shoulder_valid]))
    else:
        fallback_center = np.zeros(3, dtype=np.float32)
        fallback_scale = 1.0

    centers[~shoulder_valid] = fallback_center
    scales[~shoulder_valid] = fallback_scale
    scales = np.maximum(scales, 1e-6)

    normalized = (values - centers[:, None, :]) / scales[:, None, None]
    normalized = np.clip(normalized, clip_min, clip_max)

    sample_indices = np.rint(np.linspace(0, len(frames) - 1, target_frames)).astype(int)
    sampled = normalized[sample_indices]
    sampled_presence = presence[sample_indices]
    sampled_shoulder_valid = shoulder_valid[sample_indices]

    coordinate_features = np.concatenate(
        [
            sampled[:, :42, :].reshape(target_frames, -1),
            sampled[:, pose_start:lip_start, :2].reshape(target_frames, -1),
            sampled[:, lip_start:, :2].reshape(target_frames, -1),
        ],
        axis=1,
    )
    detection_features = np.column_stack(
        [
            sampled_presence[:, :21].mean(axis=1),
            sampled_presence[:, 21:42].mean(axis=1),
            sampled_presence[:, pose_start:lip_start].mean(axis=1),
            sampled_presence[:, lip_start:].mean(axis=1),
            sampled_shoulder_valid.astype(np.float32),
        ]
    )
    features = np.concatenate([coordinate_features, detection_features], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=clip_max, neginf=clip_min).astype(np.float32)
    selected_feature_count = int(
        (feature_configuration or {}).get("selected_feature_count", BASE_FEATURE_COUNT)
    )
    if selected_feature_count == ENHANCED_FEATURE_COUNT:
        features = build_motion_geometry_features(features)
    elif selected_feature_count != BASE_FEATURE_COUNT:
        raise ValueError(f"Unsupported selected feature count: {selected_feature_count}.")
    return features[None, ...], quality
