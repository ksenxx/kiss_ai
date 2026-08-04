# IOAI 2026 Home Task 1 — Audio Classifier

Kaggle competition: `ioai-2026-ai-models-track-practice-task-1`
Kernel: `ksenxx/ioai-2026-task1-ast-29-classes`

## Task

Extend the provided 16-class Audio Spectrogram Transformer checkpoint to 29 classes
(16 old + 13 new) in a single forward pass, using only the competition data and the
competition checkpoint, with roughly ten minutes of single-GPU training.
Metric: `0.5 * Acc_old + 0.5 * Acc_new`.

## Solution (`script.py`)

1. Decode all 1,283 clips once (sample rates range from 8 kHz to 384 kHz), resample to
   16 kHz, and compute the Kaldi log-mel filterbank the AST feature extractor uses
   (`htk_compat=True, window_type="hanning", num_mel_bins=128, dither=0.0,
   frame_shift=10`), keeping up to 30 s per clip in an fp16 cache. Normalisation is
   `(x - (-4.2677393)) / (2 * 4.5689974)`, taken from the checkpoint's
   `preprocessor_config.json`.
2. Grow the classifier from 16 to 29 rows: the 16 old rows (weights and bias) are copied
   verbatim, the 13 new rows are initialised from norm-matched class-mean prototypes
   computed with the frozen encoder.
3. Fine-tune the whole network for 10 epochs (~9.5 min on a T4): random 10.24 s crops,
   SpecAugment (one frequency mask ≤ 24 bins, one time mask ≤ 192 frames), Adam with
   `betas=(0.95, 0.999)`, `weight_decay=5e-7`, encoder LR 6e-6, head LR 1.2e-4, cosine
   decay, fp16, and a class-weighted cross entropy with `w_c ∝ sqrt(q_c / p_c)` where
   `q_c` is the prior implied by the 50/50 metric.
4. Average the weights of the last three epochs, predict by averaging the softmax over
   up to four sliding 10.24 s windows, and take the argmax.

## Leaderboard experiments

| Version | Change | Public score |
|---|---|---|
| 3 | 85 % split, argmax | 0.80599 |
| 5 | balanced assignment (≤ 13 clips per class) | 0.74116 |
| 7 | all 920 clips, argmax | **0.80815** |

Version 5 tested the hypothesis that the 363-clip evaluation set is class balanced,
which the prediction histogram seemed to support (202 old / 161 new versus
`363 * 16/29 = 200/163`). The leaderboard rejected it decisively: forcing a balanced
labelling cost 6.5 points, so the evaluation set follows the training class proportions.
`balanced_assign` and `prior_match` remain in the script as printed diagnostics.
