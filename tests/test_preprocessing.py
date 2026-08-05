from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from signlearn.preprocessing import (
    LandmarkFrame,
    build_motion_geometry_features,
    preprocess_landmark_frames,
)

ROOT = Path(__file__).resolve().parents[1]


def complete_frame(offset: float = 0.0) -> LandmarkFrame:
    left = np.full((21, 3), (0.35 + offset, 0.45, 0.0), dtype=np.float32)
    right = np.full((21, 3), (0.65 - offset, 0.45, 0.0), dtype=np.float32)
    pose = np.full((33, 3), (0.50, 0.55, 0.0), dtype=np.float32)
    pose[11] = (0.40, 0.50, 0.0)
    pose[12] = (0.60, 0.50, 0.0)
    face = np.full((478, 3), (0.50, 0.30, 0.0), dtype=np.float32)
    return LandmarkFrame(left, right, pose, face)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = json.loads(
            (ROOT / "models" / "temporal_configuration.json").read_text(encoding="utf-8")
        )
        cls.feature_configuration = json.loads(
            (ROOT / "models" / "feature_configuration.json").read_text(encoding="utf-8")
        )

    def test_output_shape_normalization_and_detection_features(self):
        frames = [complete_frame(index / 1000) for index in range(60)]
        features, quality = preprocess_landmark_frames(
            frames, self.configuration, 2.0, self.feature_configuration
        )

        self.assertEqual(features.shape, (1, 48, 492))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(quality["quality_passed"])
        np.testing.assert_allclose(features[0, :, 128], -0.5, atol=1e-5)
        np.testing.assert_allclose(features[0, :, 130], 0.5, atol=1e-5)
        np.testing.assert_allclose(features[0, :, 184:189], 1.0, atol=1e-6)
        np.testing.assert_allclose(features[0, 0, 189:477], 0.0, atol=1e-6)

    def test_missing_landmarks_fail_quality_gate_without_nan_output(self):
        def blank(count: int) -> np.ndarray:
            return np.full((count, 3), np.nan, dtype=np.float32)

        frames = [LandmarkFrame(blank(21), blank(21), blank(33), blank(478)) for _ in range(30)]
        features, quality = preprocess_landmark_frames(
            frames, self.configuration, 1.0, self.feature_configuration
        )

        self.assertFalse(quality["quality_passed"])
        self.assertGreaterEqual(len(quality["issues"]), 3)
        self.assertTrue(np.isfinite(features).all())

    def test_motion_masking_and_geometry_contract(self):
        base = np.zeros((48, 189), dtype=np.float32)
        base[:, 184:189] = 1.0
        base[1:, 0] = np.arange(1, 48, dtype=np.float32)
        enhanced = build_motion_geometry_features(base)

        self.assertEqual(enhanced.shape, (48, 492))
        np.testing.assert_allclose(enhanced[0, 189:477], 0.0, atol=1e-7)
        np.testing.assert_allclose(enhanced[1:, 189], 1.0, atol=1e-7)
        np.testing.assert_allclose(enhanced[1, 333], 1.0, atol=1e-7)
        self.assertTrue(np.isfinite(enhanced).all())

    def test_feature_name_contract(self):
        names = json.loads((ROOT / "models" / "frame_feature_names.json").read_text())
        self.assertEqual(len(names), 492)
        self.assertEqual(
            names[184:189],
            [
                "left_hand_detection_ratio",
                "right_hand_detection_ratio",
                "pose_detection_ratio",
                "lip_detection_ratio",
                "shoulder_reference_valid",
            ],
        )
        self.assertEqual(names[189], f"velocity_{names[0]}")
        self.assertEqual(names[333], f"acceleration_{names[0]}")
        self.assertEqual(names[-1], "right_fingertip_to_mouth_min_distance")


if __name__ == "__main__":
    unittest.main()
