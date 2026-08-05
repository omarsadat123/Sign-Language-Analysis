from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"
MODELS = ROOT / "models"


class EvaluationEvidenceTests(unittest.TestCase):
    def test_final_audit_evidence_is_self_consistent(self):
        decision = json.loads((RESULTS / "final_audit_decision.json").read_text(encoding="utf-8"))
        manifest = json.loads((MODELS / "deployment_manifest.json").read_text(encoding="utf-8"))
        with (RESULTS / "final_holdout_comparison.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = {(row["model"], row["view"]): row for row in csv.DictReader(stream)}

        expected_rows = {
            ("production_189", "original"): decision["production_original"],
            ("production_189", "mirrored"): decision["production_mirrored"],
            ("candidate_492", "original"): decision["candidate_original"],
            ("candidate_492", "mirrored"): decision["candidate_mirrored"],
        }
        self.assertEqual(set(rows), set(expected_rows))
        for key, expected_metrics in expected_rows.items():
            for metric in ("accuracy", "macro_f1", "top5_accuracy"):
                self.assertAlmostEqual(float(rows[key][metric]), expected_metrics[metric])

        accuracy_gain = (
            decision["candidate_original"]["accuracy"] - decision["production_original"]["accuracy"]
        )
        macro_f1_gain = (
            decision["candidate_original"]["macro_f1"] - decision["production_original"]["macro_f1"]
        )
        mirrored_gain = (
            decision["candidate_mirrored"]["macro_f1"] - decision["production_mirrored"]["macro_f1"]
        )
        self.assertAlmostEqual(accuracy_gain, decision["accuracy_gain"])
        self.assertAlmostEqual(macro_f1_gain, decision["macro_f1_gain"])
        self.assertAlmostEqual(mirrored_gain, decision["mirrored_macro_f1_gain"])

        rules = decision["frozen_rules"]
        self.assertGreaterEqual(macro_f1_gain, rules["minimum_test_macro_f1_gain"])
        self.assertGreaterEqual(
            decision["candidate_original"]["accuracy"],
            decision["production_original"]["accuracy"] - rules["test_accuracy_tolerance"],
        )
        self.assertGreaterEqual(
            decision["candidate_mirrored"]["macro_f1"],
            decision["production_mirrored"]["macro_f1"] - rules["test_mirror_f1_tolerance"],
        )
        self.assertTrue(decision["approve_for_app_integration"])
        self.assertFalse(decision["training_performed"])
        self.assertEqual(decision["external_test_status"], "WLASL not loaded or used")

        current = manifest["internal_holdout_metrics"]
        for metric in ("accuracy", "macro_f1", "top5_accuracy"):
            self.assertAlmostEqual(current[metric], decision["candidate_original"][metric])
        self.assertEqual(
            manifest["model_sha256"].lower(), decision["candidate_model_sha256"].lower()
        )

        selective = next(
            item for item in decision["selective_metrics"] if item["model"] == "candidate_492"
        )
        self.assertAlmostEqual(selective["threshold"], manifest["confidence_threshold"])
        self.assertAlmostEqual(selective["coverage"], current["coverage_at_validation_threshold"])
        self.assertAlmostEqual(
            selective["accepted_accuracy"],
            current["accepted_accuracy_at_validation_threshold"],
        )


if __name__ == "__main__":
    unittest.main()
