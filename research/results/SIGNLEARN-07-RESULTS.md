# SignLearn-07: Motion and geometry feature ablation

## Deployment decision

**Keep the current 189-feature mirror-robust production model.**

The 492-feature motion/geometry model improved over its matched 189-feature control, so the
feature idea is promising. However, it did not outperform the model already deployed in the
webcam application. Replacing the production model would therefore reduce measured performance.

## Results

| Model | Evaluation | Accuracy | Macro F1 | Top-5 accuracy |
|---|---|---:|---:|---:|
| SignLearn-07 control (189 features) | Validation | 72.93% | 72.47% | 91.71% |
| SignLearn-07 motion/geometry (492 features) | Validation | **73.18%** | **72.85%** | **92.61%** |
| SignLearn-07 control (189 features) | Google holdout | 70.45% | 70.41% | 89.58% |
| SignLearn-07 motion/geometry (492 features) | Google holdout | **70.68%** | **70.48%** | **89.91%** |
| Current deployed mirror-robust model | Google holdout | **71.13%** | **71.04%** | **90.03%** |

Against its controlled baseline, the 492-feature model gained approximately 0.23 percentage
points of holdout accuracy and 0.08 points of macro F1. Against the current production model, it
lost approximately 0.45 points of accuracy and 0.55 points of macro F1.

## Selective prediction

At the experiment-selected confidence threshold of 0.63, the motion/geometry model covered
73.57% of the Google holdout and achieved 85.98% accuracy on accepted predictions (1,962 accepted
samples). This remains useful evidence for the abstention design, but does not justify changing
the deployed classifier.

## Per-class effects on validation

The largest F1 improvements included `wake`, `gift`, `uncle`, `find`, and `napkin`. The largest
regressions included `awake`, `brother`, `yesterday`, `who`, and `lips`. This suggests that motion
features help some temporally distinctive signs but can also amplify noise or redundancy for
other classes.

## Interpretation

This is a valid negative deployment result: the controlled ablation supports the hypothesis that
motion/geometry features carry signal, while the comparison with the stronger production model
shows that adding features alone is not sufficient. WLASL was not used for training or model
selection and remains available for external distribution-shift testing.

## Recommended next experiment

Initialize a 492-feature model from the current mirror-robust checkpoint: copy the learned weights
for the original 189 inputs, initialize only the new motion/geometry input weights to zero, then
fine-tune with a low learning rate. This tests whether the new information can improve the stronger
model without discarding its existing representation. Select the checkpoint using validation data;
avoid repeatedly optimizing decisions against the Google holdout.
