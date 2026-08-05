from __future__ import annotations

import hashlib
import json
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


class BundleContractTests(unittest.TestCase):
    def test_labels_manifest_and_network_agree(self):
        labels = json.loads((MODELS / "label_map.json").read_text(encoding="utf-8"))
        manifest = json.loads((MODELS / "deployment_manifest.json").read_text(encoding="utf-8"))
        with zipfile.ZipFile(MODELS / "signlearn_model.keras") as archive:
            network = json.loads(archive.read("config.json"))

        layers = network["config"]["layers"]
        input_layer = next(layer for layer in layers if layer["class_name"] == "InputLayer")
        output_layer = next(
            layer for layer in layers if layer["config"]["name"] == "sign_probabilities"
        )

        self.assertEqual(sorted(labels.values()), list(range(50)))
        self.assertEqual(manifest["class_count"], 50)
        feature_configuration = json.loads(
            (MODELS / "feature_configuration.json").read_text(encoding="utf-8")
        )
        feature_names = json.loads((MODELS / "frame_feature_names.json").read_text())
        self.assertEqual(manifest["input_shape"], [1, 48, 492])
        self.assertEqual(manifest["mirror_augmentation_mode"], "reflect_and_swap_hands")
        self.assertEqual(manifest["confidence_threshold"], 0.60)
        self.assertEqual(manifest["operational_confidence_threshold"], 0.71)
        self.assertEqual(feature_configuration["selected_feature_count"], 492)
        self.assertEqual(len(feature_names), 492)
        self.assertEqual(input_layer["config"]["batch_shape"], [None, 48, 492])
        self.assertEqual(output_layer["config"]["units"], 50)
        self.assertEqual(output_layer["config"]["activation"], "softmax")

    def test_model_checksum(self):
        digest = hashlib.sha256((MODELS / "signlearn_model.keras").read_bytes()).hexdigest().upper()
        self.assertEqual(
            digest,
            "5FCCE7F542D60EAAF36F4F78978A00125A520346A75DC8F71571E465FA3D11D5",
        )


if __name__ == "__main__":
    unittest.main()
