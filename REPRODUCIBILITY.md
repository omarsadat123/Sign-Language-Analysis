# Reproducibility guide

## Frozen deployment artifacts

The `models/` directory contains:

- `signlearn_model.keras` — approved TensorFlow/Keras classifier
- `temporal_configuration.json` — landmark selection, sampling, normalization, clipping
- `feature_configuration.json` — motion and geometry feature contract
- `frame_feature_names.json` — ordered 492-feature schema
- `label_map.json` — 50-class mapping
- `deployment_manifest.json` — thresholds, metrics, scope, hashes
- `camera_orientation_policy.json` — mirror and confidence policy
- `mirror_validation_audit.json` — normal/mirrored comparison evidence

Run `python -m unittest discover -s tests -v` to verify the executable contract. Compare artifact
hashes with `CHECKSUMS.sha256`.

## Environment

The tested target is 64-bit Python 3.11 with TensorFlow 2.20.0. Exact deployment dependencies are
pinned in `requirements.txt`. A CPU is sufficient for inference; a GPU is recommended for training.

## Research sequence

1. **SignLearn-06:** fine-tune the temporal model with 50% reflection plus left/right hand-channel
   swapping; choose orientation and confidence policy on validation signers.
2. **SignLearn-07:** compare a 189-feature control with a 492-feature motion/geometry model under
   matched training. The enhanced model beat its matched control but not production, so deployment
   was rejected.
3. **SignLearn-08:** initialize a new 303-feature projection at zero and copy all learned production
   weights. The notebook exposed a mismatch between loss-based checkpointing and macro-F1 selection.
4. **SignLearn-08B:** evaluate validation macro F1 every epoch and save the correct checkpoint. The
   best checkpoint occurred at epoch 2 and passed validation gates.
5. **SignLearn-09:** perform no training; compare frozen production and candidate checkpoints using
   predeclared internal-holdout rules. The candidate passed and was integrated.

## Critical invariants

- Split participants must be disjoint.
- Feature order must match `frame_feature_names.json` exactly.
- Transfer initialization must reproduce source probabilities with maximum absolute error `<1e-5`;
  the recorded experiment obtained `0.0`.
- First-frame velocity and acceleration must be zero.
- Motion and geometry must be masked when required landmarks are missing.
- Checkpoint selection and promotion must use validation macro F1.
- Confidence thresholds must be selected on validation, not holdout or WLASL.

## Reported final numbers

| Evaluation | Accuracy | Macro F1 | Top-5 |
|---|---:|---:|---:|
| Validation, original | 73.54% | 73.34% | 93.11% |
| Validation, mirrored | 70.55% | 70.48% | 91.13% |
| Internal holdout, original | 71.69% | 71.60% | 89.69% |
| Internal holdout, mirrored | 70.72% | 70.98% | 88.53% |

At threshold 0.60, validation coverage/accepted accuracy were 75.67%/85.47%; internal-holdout
coverage/accepted accuracy were 77.50%/84.03%.

## What cannot be reproduced from this repository alone

- Training without obtaining the competition data under its terms
- A pristine external evaluation, because none was completed
- Webcam-user performance, because no consented external study has been completed
- Exact training-time hardware behavior across Kaggle image updates

The paper and model card distinguish these gaps from completed evidence.
