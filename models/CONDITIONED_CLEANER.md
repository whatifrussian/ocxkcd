# Conditioned cleaner v1

Selected color/transparency-augmented experiment, seed 2026, epoch 60
(80 training epochs). Threshold 0.3, U-Net width 16, five input channels:
visible RGB on white, original alpha, and automatic proposal coverage.
The checkpoint contains weights and provenance, not local training paths.

Training used 120 previously reviewed brush-mask crops from 20 source images:
85 training, 16 validation, 19 test crops. Whole article/comic groups were
separated; test groups were what-if-160 and xkcd-3270. These are historical
reviewed holdouts, not a new blind test. Overlapping annotations contain some
conflicting labels; source annotations were not silently rewritten.

The selected training dataset manifest SHA-256 is
`a9fa94e3f02a9d1da5a0f1791fc04a3e734bd484a5f9b11a1cb1e30dc1112cc0`.

Inference runs on automatic OCR-line crops with 26-pixel context, at native
resolution with overlapping 256-pixel tiles. Predictions are restricted to
within eight pixels of the proposed mask. Existing RGBA fills are transferred
to selected pixels; new coverage uses the nearest proposed fill. Text
transcription and geometry are unchanged. Do not substitute whole-page
inference: it was tested and performed worse.

On 19 held-out crops aligned to actual exports, mean mask Dice was 0.9564,
versus 0.9337 for the baseline. The selected report contains 40 full images.
Remaining issues include occasional artwork damage (notably a head in
xkcd-3261), incomplete CRASH cleanup, and remnants near borders. This model
does not reconstruct artwork hidden behind text.

To package compatible trained weights, run
`python scripts/package-cleaner.py /path/to/selected-checkpoint.pt`.
This keeps both Docker and wheel copies in sync, removing machine-specific
training arguments without changing learned parameters.
