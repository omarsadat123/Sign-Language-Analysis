# SignLearn: Transfer-Initialized Motion Fusion and Selective Prediction for a 50-Class Isolated ASL Practice Prototype

**Draft manuscript — not peer reviewed**  
**Author note:** *Identity and affiliation omitted from this working draft*  
**Date:** 5 August 2026

## Abstract

Isolated sign language recognition (ISLR) is a bounded classification problem that can support
dictionary retrieval or carefully framed practice tools, but it must not be conflated with
continuous sign-language recognition or translation. This paper presents SignLearn, a local,
quality-gated research prototype that recognizes 50 isolated American Sign Language (ASL) labels
from short webcam videos. The system transforms MediaPipe Holistic landmarks into 48-frame
sequences containing 189 normalized landmark and detection features, 144 velocity features, 144
acceleration features, and 15 masked geometry measurements. A temporal classifier combines two
bidirectional long short-term memory layers with multi-head self-attention and confidence-based
abstention. Development used signer-disjoint partitions containing 15 training, three validation,
and three internal-holdout signers. An initial motion-feature model improved over a matched control
but underperformed the stronger production model. We therefore introduced a transfer-initialized
fusion branch: the existing 189-feature projection and downstream network were copied, while the
new 303-feature projection was initialized to zero, reproducing source probabilities exactly before
fine-tuning. Correcting checkpoint selection to use validation macro F1 yielded a candidate that
passed predeclared validation and comparative-holdout gates. Relative to the prior production model,
the candidate increased internal-holdout accuracy from 71.13% to 71.69%, macro F1 from 71.04% to
71.60%, and mirrored-view macro F1 from 69.88% to 70.98%; top-5 accuracy decreased from 90.03% to
89.69%. At the validation-selected confidence threshold of 0.60, coverage was 77.50% and accepted
accuracy was 84.03% on the internal holdout. The application uses a provisional stricter threshold
of 0.71 pending webcam calibration. These results concern a small landmark dataset and are not
evidence of ASL translation quality, webcam generalization, subgroup equity, or accessibility
impact. The code, tests, manifests, notebooks, model card, and frozen audit evidence are released to
support reproducibility and critical review.

**Keywords:** isolated sign language recognition; American Sign Language; MediaPipe; bidirectional
LSTM; self-attention; motion features; selective prediction; signer-independent evaluation

## 1. Introduction

Signed languages are natural languages expressed through manual and non-manual articulators. A
computational system that assigns a short clip to one vocabulary label solves a substantially
narrower problem than recognizing continuous signing, modeling linguistic structure, or translating
between languages. Bragg et al. argue that successful sign-language technology requires expertise
across machine learning, linguistics, human-computer interaction, accessibility, and Deaf culture,
and warn against isolated technical framing [1]. We adopt that boundary explicitly: SignLearn is a
research and portfolio prototype for low-stakes practice feedback over 50 labels, not an interpreter
or translator.

Recent datasets illustrate both the progress and difficulty of ISLR. WLASL introduced more than
2,000 word-level signs from over 100 signers and compared appearance- and pose-based methods [2].
ASL Citizen later released a consented, community-sourced dataset with 83,399 videos, 2,731 signs,
and 52 signers, emphasizing dictionary retrieval and evaluation on unseen users [3]. SignLearn uses
the separate Google Isolated Sign Language Recognition competition resource [4], restricted to 50
classes and represented as landmark sequences. Neither WLASL nor ASL Citizen was used for training
or model selection in this project.

The project addresses four practical questions:

1. Can temporal modeling improve over a frame-flattened baseline under signer-independent splits?
2. Do pose, lip, motion, and geometry features provide measurable complementary information?
3. Can horizontal-reflection augmentation improve robustness to camera orientation without
   degrading the calibrated view?
4. Can a stronger production model absorb new motion features without discarding its learned
   representation?

The fourth question motivated the final contribution. A naïvely trained 492-feature network slightly
outperformed its matched control but remained weaker than production. Rather than presenting that
negative result as an improvement, we rejected it and constructed a transfer-initialized motion
branch whose initial output was mathematically and numerically identical to the deployed model. This
controlled the cost of adding features and exposed a second methodological issue: checkpointing by
validation loss was inconsistent with promotion by macro F1. A corrected macro-F1 checkpoint
produced the final candidate.

The principal contributions are:

- an explicit 492-feature, detection-masked motion and geometry representation;
- a transfer initialization that reproduces a stronger 189-feature model exactly at epoch zero;
- signer-independent, normal/mirrored, selective-prediction, and per-class evaluation;
- an audit trail containing failed ablations and checkpoint-selection corrections;
- a local quality-aware Gradio application with frozen preprocessing and confidence policies; and
- documentation that separates completed dataset evidence from untested webcam and accessibility
  claims.

## 2. Related work

### 2.1 Isolated sign language recognition

ISLR maps a bounded video segment to a lexical label. WLASL demonstrated the scale and variability
of word-level recognition and introduced pose-based temporal graph modeling alongside appearance
baselines [2]. ASL Citizen broadened ISLR toward community-sourced, consented, in-the-wild videos and
evaluated entirely on users excluded from training and validation [3]. These studies support two
choices in SignLearn: evaluation by signer rather than random sequence, and reporting top-k metrics
alongside top-1. They also underscore why a 50-class classroom prototype cannot support broad claims.

### 2.2 Landmark perception

MediaPipe Hands combines palm detection with hand-landmark estimation for real-time on-device
tracking [5]. MediaPipe Holistic integrates pose, face, and both hand components into a semantically
consistent topology [6], and its video API expects monotonically increasing timestamps [7].
Landmarks reduce the dimensionality and privacy exposure of raw video, but they are not neutral:
tracking errors, occlusion, camera conditions, skin appearance, motion, and domain differences can
propagate into the classifier. SignLearn therefore reports detection ratios and rejects visibly
unsuitable sequences before neural inference.

### 2.3 Temporal modeling and attention

Long short-term memory networks were designed to address long-range dependency and gradient issues
in recurrent learning [8]. Bidirectional recurrence processes a sequence in forward and reverse
directions [9], which is appropriate here because an isolated clip is available in full before
classification. Self-attention provides learned interactions between temporal positions [10].
SignLearn uses recurrence for ordered motion context and a compact attention block for frame-to-frame
relationships rather than adopting a large video transformer.

### 2.4 Selective prediction

Selective classification introduces a reject option that trades coverage for lower risk on accepted
examples [11]. SelectiveNet extends the concept through joint optimization [12]. SignLearn uses the
simpler post-hoc maximum-softmax response because the classifier was already trained and the project
focus was deployment behavior. A validation-selected threshold is reported separately from the
provisional operational threshold. Abstention is a product behavior, not a guarantee of calibrated
risk.

## 3. Data and evaluation design

### 3.1 Task and subset

The source resource is the Google Isolated Sign Language Recognition Kaggle competition [4]. The
project selected 50 classes and cached sequences as 48 frames of 189 base features. The frozen label
map is distributed with the model. Raw competition data is not redistributed.

### 3.2 Signer-independent partition

Sequences were divided by participant: 15 training signers, three validation signers, and three
internal-holdout signers. The participant sets were asserted to be pairwise disjoint. Validation
controlled architecture, features, checkpoint, camera policy, and confidence threshold. A final
candidate was compared with production on the holdout only after passing the validation gate.

The holdout had already appeared in earlier project reports. Repeated reporting weakens its status as
an untouched test, even if the final candidate was not trained on it. We therefore call it an
*internal comparative holdout*, not an external test. WLASL was intentionally left unused; no claim
is made that this alone establishes external validity.

### 3.3 Metrics

We report top-1 accuracy, unweighted macro F1, top-5 accuracy, selective coverage, accepted accuracy,
and per-class F1 [13]. Macro F1 gives each of the 50 labels equal weight. For probabilities
$p_i \in \mathbb{R}^{50}$ and threshold $\tau$, the accepted set is

$$A_\tau = \{i: \max_c p_{ic} \ge \tau\}.$$

Coverage is $|A_\tau|/N$ and accepted accuracy is the fraction correct within $A_\tau$. The threshold
was selected on validation signers to reach at least 85% accepted accuracy with at least 10%
coverage, choosing the smallest eligible threshold.

## 4. Method

### 4.1 Landmark selection and normalization

For each frame, SignLearn selects 21 landmarks from each hand, nine pose landmarks, and twenty lip
landmarks. Hand coordinates retain $(x,y,z)$; pose and lip coordinates use $(x,y)$. Five additional
features represent left-hand, right-hand, pose, lip, and shoulder-reference detection. Thus the base
feature count is

$$2(21 \times 3) + (9 \times 2) + (20 \times 2) + 5 = 189.$$

Coordinates are normalized using the shoulder midpoint $m_t$ and shoulder distance $s_t$:

$$\tilde{x}_{t,j} = \operatorname{clip}\left(\frac{x_{t,j}-m_t}{\max(s_t,10^{-6})},-5,5\right).$$

When shoulder references are unavailable, median valid center and scale values from the clip are
used; if none exist, zero center and unit scale are used. The complete clip is uniformly sampled to
48 nearest frames. Missing coordinates become zero only after detection indicators are retained.

### 4.2 Motion and geometry

Motion operates on the first 144 coordinates: 126 hand dimensions and 18 selected pose dimensions.
For frame $t>0$,

$$v_t = \tilde{x}_t - \tilde{x}_{t-1}, \qquad a_t = v_t - v_{t-1}.$$

The first velocity and acceleration are zero. Each hand or pose group is zeroed unless it is detected
in the necessary adjacent frames; $v_t$ and $a_t$ are clipped to $[-5,5]$. This produces 144 velocity
and 144 acceleration features.

Fifteen geometry features include wrist-to-wrist distance; wrist-to-mouth, nose, and corresponding
shoulder distances; mean fingertip spread; thumb-to-index distance; hand-wrist to pose-wrist
distance; and minimum fingertip-to-mouth distance. A geometry value is zero unless every landmark
group required for that measurement is detected. Values are clipped to $[0,10]$. The final frame
representation contains $189+144+144+15=492$ features.

### 4.3 Architecture

The production architecture maps each frame to 128 dimensions, applies layer normalization [14] and
spatial dropout [15], then processes the sequence with bidirectional LSTMs containing 96 and 64 units
per direction. Four-head self-attention with key dimension 32 operates on the second recurrent
sequence and is added through a normalized residual connection. Global average and maximum pooling
are concatenated, followed by a 192-unit ReLU layer, 0.35 dropout, and a 50-class softmax.

### 4.4 Transfer-initialized fusion

Directly training a 492-feature model discarded the stronger model's representation. The final
network instead splits each enhanced frame into base features $b_t\in\mathbb{R}^{189}$ and new
features $m_t\in\mathbb{R}^{303}$:

$$h_t = \operatorname{ReLU}(W_b b_t + q + W_m m_t).$$

$W_b$ and bias $q$ are copied from the deployed frame projection; $W_m$ is initialized to zero and
has no bias. Therefore $h_t$ is initially identical to the production activation for every input.
All downstream weights are copied. The measured maximum absolute probability difference over the
initialization probe was exactly 0.0.

Training has two stages. For five warm-up epochs, only $W_m$ is trainable with Adam [16] at
$3\times10^{-4}$. The complete model is then unfrozen and fine-tuned at $5\times10^{-5}$ for at most
24 epochs. Training examples are mirrored with probability 0.5 by reflecting horizontal coordinate
channels and swapping left/right hand features and detection indicators.

### 4.5 Checkpoint and promotion policy

An initial implementation saved the lowest validation-loss checkpoint while promoting by macro F1.
This inconsistency retained an epoch with 73.04% validation accuracy even though a later epoch
reached 73.58%. The corrected experiment computed macro F1 after every epoch, saved the best
macro-F1 checkpoint, and stopped after seven non-improving epochs. The selected checkpoint occurred
at epoch 2.

Promotion from validation required: (i) at least +0.20 percentage points macro F1; (ii) accuracy no
more than 0.10 points lower; and (iii) mirrored macro F1 no more than 0.50 points lower. Final
integration required non-negative internal-holdout macro-F1 gain, accuracy no more than 0.10 points
lower, and mirrored macro F1 no more than 0.50 points lower. These rules were frozen before the
respective predictions.

## 5. Experiments and results

### 5.1 Development progression

| Experiment | Principal finding |
|---|---|
| MLP baseline | 63.96% internal-holdout macro F1 |
| Temporal BiLSTM/self-attention | approximately 70.93% macro F1 |
| Hands-only ablation | 61.58% validation macro F1 |
| Hands + pose | 67.01% validation macro F1 |
| Hands + pose + lips | 70.22% validation macro F1 |
| Mirror fine-tuning | normal validation macro F1 73.00%; mirrored 70.00% |
| Direct 492-feature ablation | beat matched control but remained below production; rejected |
| Transfer model, loss checkpoint | below production; selection mismatch diagnosed |
| Transfer model, macro-F1 checkpoint | passed validation and final comparative audit |

This sequence shows that feature value depends on initialization and selection protocol, not merely
feature count.

### 5.2 Validation results

| Model/view | Accuracy | Macro F1 | Top-5 |
|---|---:|---:|---:|
| Production 189, original | 73.25% | 73.00% | **93.40%** |
| Candidate 492, original | **73.54%** | **73.34%** | 93.11% |
| Production 189, mirrored | 70.15% | 70.00% | **91.24%** |
| Candidate 492, mirrored | **70.55%** | **70.48%** | 91.13% |

The candidate gained 0.33 macro-F1 points in the calibrated view and 0.48 points in the synthetic
mirrored view while losing 0.29 top-5 points in the original view. At $\tau=0.60$, validation
coverage was 75.67% and accepted accuracy was 85.47%.

### 5.3 Comparative internal holdout

| Model/view | Accuracy | Macro F1 | Top-5 |
|---|---:|---:|---:|
| Production 189, original | 71.13% | 71.04% | **90.03%** |
| Candidate 492, original | **71.69%** | **71.60%** | 89.69% |
| Production 189, mirrored | 69.74% | 69.88% | **88.75%** |
| Candidate 492, mirrored | **70.72%** | **70.98%** | 88.53% |

The final candidate improved accuracy by 0.56 points, macro F1 by 0.56 points, and mirrored macro F1
by 1.10 points. Top-5 accuracy decreased by 0.34 points in the original view and 0.22 in the mirrored
view. At their respective validation-selected thresholds, production achieved 83.68% accepted
accuracy at 77.43% coverage, while the candidate achieved 84.03% at 77.50% coverage.

Per-class effects were heterogeneous. The largest holdout F1 improvements included `pajamas`
(+6.03 points), `many` (+5.33), `wet` (+4.89), `listen` (+4.35), and `airplane` (+3.95). The largest
regressions included `mouse` (-6.86), `wake` (-6.10), and `bird` (-5.36). These variations caution
against interpreting a small aggregate gain as uniform improvement.

### 5.4 Deployment smoke test

The bundled 492-feature model passed preprocessing and bundle-contract tests, a frozen-evaluation
evidence test, and a local TensorFlow load/inference check. Evaluation recordings are not
redistributed; reviewers can test the application with a recording they own or are authorized to
use. This is an engineering check, not an accuracy estimate.

## 6. Discussion

The central empirical result is modest: motion fusion improved macro F1 by approximately half a
percentage point. Its methodological value is larger. The direct enhanced model failed against the
stronger production checkpoint, demonstrating that a positive controlled ablation does not imply a
deployment improvement. Zero-initialized transfer fusion allowed the project to test new inputs
without beginning from a weaker random solution. Exact initial equivalence also converted a vague
transfer-learning claim into a testable invariant.

Checkpoint selection mattered. Validation loss and macro F1 encode different preferences;
cross-entropy is sensitive to the full probability distribution, whereas macro F1 depends on hard
class decisions and weights each class equally. Selecting by one and promoting by the other can
silently invalidate a comparison. The corrected callback did not guarantee generalization, but it
aligned implementation with the declared objective.

The decline in top-5 accuracy is also informative. Motion fusion improved the most likely class and
macro balance while slightly worsening whether the true class appeared anywhere in the top five.
Applications centered on dictionary retrieval might prioritize recall-at-k, as in ASL Citizen [3],
and could reject this trade-off. SignLearn's UI presents top three predictions but treats top-1 with
abstention as the primary behavior. The appropriate metric depends on the user task and should be
co-designed rather than selected after observing favorable numbers.

The selective policy improved accepted accuracy only slightly and remained below the validation
target on holdout. The operational threshold of 0.71 is intentionally provisional and was not tuned
on the holdout. A separate webcam-development set is required to measure its actual risk-coverage
curve. Maximum softmax probability is convenient but is not calibrated correctness; future work
should compare temperature scaling, conformal or risk-controlling procedures, and integrated reject
models without reusing the final evaluation set.

## 7. Responsible use, limitations, and threats to validity

### 7.1 Construct validity

The labels represent isolated examples, not continuous language competence. The model cannot
evaluate grammar, naturalness, fluency, regional variation, or semantic appropriateness. A predicted
label is not proof that a person signed correctly.

### 7.2 Internal validity

Signer-disjoint splits reduce one leakage mechanism, but data cleaning, class selection, repeated
experimentation, and checkpoint choices can still overfit the development setting. The holdout was
reported in previous iterations and is not pristine. The small 0.56-point gain has no confidence
interval or repeated-seed analysis; sampling and optimization variability may be comparable.

### 7.3 External validity

No external webcam study was completed. The webcam pipeline uses a current MediaPipe Holistic task,
whereas the training cache may reflect a different extraction environment. Lighting, framing,
background, camera optics, movement speed, occlusion, clothing, skin appearance, mobility, and
signing variation may shift performance. WLASL was kept unused, but simply reserving a dataset is
not equivalent to completing an external evaluation.

### 7.4 Human and societal risks

Sign-language systems can encode hearing-centered assumptions, exclude Deaf expertise, collect
sensitive visual data, and be repurposed for assessment or surveillance [1]. SignLearn processes
locally and uses quality and confidence gates, but these are partial mitigations. Any study should be
developed with Deaf/ASL stakeholders, obtain informed consent, define withdrawal and deletion,
compensate expertise, and report tracking failures and abstentions. Consequential educational,
employment, medical, emergency, or interpreting use is outside scope.

### 7.5 Licensing and reproducibility

The repository does not redistribute the Kaggle competition dataset or MediaPipe task binary. The
code license does not automatically license third-party data, derived model weights, dependencies,
or recordings. Model-weight redistribution must be checked against the current competition terms
before public release. Reproducibility is therefore conditional on authorized data access.

## 8. Future work

The next experiment should be a pre-registered, consented study with signers absent from every
development resource. It should measure top-1 accuracy, macro F1, recall-at-k, quality-gate failure,
coverage, accepted accuracy, latency, and results by recording condition. Operational calibration
must use a separate development subset. Study questions and UI feedback should be co-designed with
Deaf/ASL stakeholders.

Technical extensions include uncertainty calibration, external distribution-shift evaluation,
motion representations robust to variable frame rate, comparison with graph or transformer models,
model compression with numerical parity tests, and better diagnostics for landmark tracking. Any
move toward continuous recognition or translation should be treated as a new problem requiring
linguistic annotation, substantially broader data, and a different evaluation design.

## 9. Conclusion

SignLearn demonstrates a disciplined path from coursework to an auditable deep-learning prototype.
A signer-independent temporal model, mirror robustness, motion and geometry features, transfer
initialization, macro-F1-aligned checkpointing, and selective prediction were connected to a local
web application with frozen contracts and tests. The final motion-fusion model modestly improved
accuracy and macro F1 while slightly reducing top-5 accuracy. More importantly, the project records
negative results and limits its claims: it is a 50-class isolated-sign prototype with no completed
external webcam validation. Releasing the code, evidence, and manuscript enables review, but useful
sign-language technology ultimately requires participatory research and evaluation beyond this
technical artifact.

## References

[1] D. Bragg et al., “Sign Language Recognition, Generation, and Translation: An Interdisciplinary
Perspective,” *ASSETS*, 2019. https://arxiv.org/abs/1908.08597

[2] D. Li, C. Rodriguez, X. Yu, and H. Li, “Word-level Deep Sign Language Recognition from Video: A
New Large-scale Dataset and Methods Comparison,” *WACV*, 2020.

[3] A. Desai et al., “ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language
Recognition,” *NeurIPS Datasets and Benchmarks*, 2023. doi:10.52202/075280-3360.

[4] Google and Kaggle, “Google — Isolated Sign Language Recognition,” 2023.
https://www.kaggle.com/competitions/asl-signs

[5] F. Zhang et al., “MediaPipe Hands: On-device Real-time Hand Tracking,” arXiv:2006.10214, 2020.

[6] I. Grishchenko and V. Bazarevsky, “MediaPipe Holistic: Simultaneous Face, Hand and Pose
Prediction, on Device,” Google Research, 2020.

[7] Google AI for Developers, “HolisticLandmarker Python API,” accessed 5 Aug. 2026.

[8] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, 9(8),
1735–1780, 1997. doi:10.1162/neco.1997.9.8.1735.

[9] TensorFlow, “tf.keras.layers.Bidirectional,” official API documentation, accessed 5 Aug. 2026.

[10] A. Vaswani et al., “Attention Is All You Need,” *NeurIPS*, 2017.

[11] Y. Geifman and R. El-Yaniv, “Selective Classification for Deep Neural Networks,” *NeurIPS*,
2017.

[12] Y. Geifman and R. El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject
Option,” *ICML*, 2019.

[13] scikit-learn developers, “Metrics and scoring: quantifying the quality of predictions,” accessed
5 Aug. 2026.

[14] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer Normalization,” arXiv:1607.06450, 2016.

[15] N. Srivastava et al., “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,”
*JMLR*, 15, 1929–1958, 2014.

[16] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” *ICLR*, 2015.

Complete machine-readable entries are provided in `references.bib`.
