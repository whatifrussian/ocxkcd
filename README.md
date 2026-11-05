# ocxkcd

`ocxkcd` runs OCR on xkcd/what-if style comic images and exports files that can be opened by `comic_editor`:

- original image copy
- cleaned image
- text-shaped mask
- `*.meta.json`
- `*.text.json`
- raw PaddleOCR detections
- HTML review report

The current release path is based on PP-OCRv5 server detection/recognition plus the cleanup/fill heuristics developed for the what-if/xkcd workflow.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[cpu,ml]"
ocxkcd process image.png -o output/image
```

For several images:

```bash
ocxkcd process ./images --recursive -o ./output
```

The result directory contains one subfolder per input image. Each subfolder contains files like:

```text
example.png
example.clean.png
example.mask.png
example.mask.preview.png
example.meta.json
example.text.json
example.paddle.json
```

Open `output/index.html` for the review report.

## Docker

Build locally:

```bash
docker build -t ocxkcd .
docker run --rm \
  -v "$PWD/images:/input:ro" \
  -v "$PWD/output:/output" \
  ocxkcd process /input --recursive -o /output
```

The Docker image is intended to be self-contained:

- official PaddleOCR server detector/recognizer models are downloaded into `/models` during image build;
- the small `ocxkcd` crop-refiner, fill-prior, and conditioned-cleaner checkpoints are copied into `/models/ocxkcd`.

Do not mount an empty directory over `/models` unless you intentionally want to replace the baked models.

### Docker example with a sibling website checkout

Assume this repository and the website repository are siblings:

```text
parent/
  ocxkcd/
  website/
```

Process two existing article upload folders while skipping already-translated `_ru` files:

```bash
cd ocxkcd
docker build -t ocxkcd:local .
mkdir -p output/website-sample

docker run --rm \
  -v "$PWD/../website/content/uploads:/uploads:ro" \
  -v "$PWD/output/website-sample:/output" \
  ocxkcd:local process \
    /uploads/006-glass-half-empty \
    /uploads/030-interplanetary-cessna \
    --exclude-glob '*_ru.*' \
    -o /output
```

Review the generated quality reports:

```text
output/website-sample/index.html
output/website-sample/_work/ocr/index.html
```

`index.html` shows the comic_editor export artifacts: original, cleaned image, mask preview, box preview, accepted detections, fill colors, and rejected OCR candidates.

`_work/ocr/index.html` shows the raw PaddleOCR detections before cleanup/export, useful when deciding whether an issue is OCR detection/recognition or cleanup/fill.

Published image name:

```text
ghcr.io/<owner>/ocxkcd
```

After pushing this repository to GitHub, the `Docker GHCR` workflow publishes:

- branch tags
- Git tags like `v0.1.0`
- commit SHA tags

## GitHub Pages quality report

The `Pages Report` workflow builds the Docker image, checks out `whatifrussian/website`,
processes a small real-image sample, and publishes the generated HTML report as a
GitHub Pages artifact for:

```text
https://ocxkcd.chtoes.li/
```

The sample currently uses:

```text
website/content/uploads/006-glass-half-empty
website/content/uploads/030-interplanetary-cessna
```

Files ending in `_ru.*` are skipped so the report is generated from source images,
not already-translated images.

Published report paths:

```text
/
/_work/ocr/index.html
```

The root report is the comic_editor export/cleanup report. The `_work/ocr` report is
the raw PaddleOCR detection report used for debugging OCR versus cleanup/fill issues.
The workflow also adds `.nojekyll` so underscore-prefixed report directories are
served correctly.

For the custom domain, configure GitHub Pages to use Actions as the Pages source and
point `ocxkcd.chtoes.li` at GitHub Pages DNS. The workflow writes `CNAME` into the
published artifact.

## GPU

CPU is the default and is adequate for batch processing. For GPU, use a Paddle/PaddleOCR image with CUDA support or extend the Dockerfile from a CUDA base image, then run:

```bash
ocxkcd process ./images -o ./output --device gpu:0
```

The cleanup refiner and fill-prior checkpoints use PyTorch if provided. Pass `--crop-refiner-device cuda` or `--fill-prior-device cuda` when running in a CUDA-capable environment.

## Model artifacts

The default cleanup now includes the approved proposal-conditioned pixel cleaner
(`conditioned-cleaner-v1.pt`, about 2 MB). It receives the original RGBA image and
automatic cleanup proposal, refines individual pixels in overlapping OCR-line
crops, and preserves the existing background-color/alpha estimates and transcription.
It does not use manually corrected masks at inference. The checkpoint is bundled
in the repository, Python wheel, and Docker image; no training checkout is needed.

Use `--cleaner legacy` to reproduce the previous cleanup pipeline, or
`--cleaner-device cuda` to run the new stage on GPU (CPU is the default).
`--cleaner-checkpoint PATH` accepts a compatible replacement checkpoint.
Install the `ml` extra when running outside Docker. Missing or incompatible
checkpoints fail explicitly rather than silently changing the cleanup method.
The legacy crop refiner remains part of proposal generation; its explicit
`--crop-refiner-combine off` setting is now respected.

Validation compared 40 full comics, including transparent/colored backgrounds,
rotated text, and tables. On 19 held-out reviewed crops the exported-mask mean
Dice increased from 0.934 to 0.956. These are a small, historical test set, not
a guarantee for unseen comics; occasional artwork damage and letter remnants
remain possible. OCR recognition and background estimation were not retrained.
See [model provenance](models/CONDITIONED_CLEANER.md) for training and inference details.

To check an existing approved 40-comic comparison report against the packaged
model without rerunning OCR:

```bash
OCXKCD_CLEANER_REFERENCE=/path/to/approved-report/manifest.json pytest tests/test_cleaner_regression.py
```

The test compares masks and cleaned images pixel-for-pixel. Normal CI runs
synthetic mask/alpha tests and both legacy and conditioned exports through
the actual `comic_editor` renderer, with CPU PyTorch installed.

There are four classes of model artifacts:

1. PaddleOCR official detector/recognizer models. These are required for OCR unless PaddleOCR is allowed to download them at first run.
2. `ocxkcd` crop-refiner checkpoint. Optional, but recommended for tighter text-shaped masks.
3. `ocxkcd` fill-prior checkpoint. Optional, but recommended for white/black/transparent fill decisions.
4. `ocxkcd` conditioned-cleaner checkpoint, enabled by default after proposal generation.

The source repository commits our small `.pt` checkpoints. It does not commit the much larger official PaddleOCR model directories. Docker builds and GitHub runners can download/cache those official PaddleOCR models.

Expected artifact layout when importing from a training/export checkout:

```text
<source-root>/_models/paddlex/official_models/PP-OCRv5_server_det
<source-root>/_models/paddlex/official_models/PP-OCRv5_server_rec
<source-root>/_text_artwork_training_v1/runs/refiner-smoke-seed1337/best.pt
<source-root>/_fill_training_v1/runs/fill-classifier-stratified-seed2026/best.pt
```

Prepare local runtime artifacts:

```bash
scripts/prepare-models.sh --copy --source-root /path/to/training-checkout
```

or use symlinks:

```bash
scripts/prepare-models.sh --symlink --source-root /path/to/training-checkout
```

Then run with the learned helpers:

```bash
ocxkcd process image.png -o output \
  --model-root ./models \
  --crop-refiner-checkpoint ./models/ocxkcd/refiner-smoke-seed1337.pt \
  --crop-refiner-combine prefer \
  --fill-prior-checkpoint ./models/ocxkcd/fill-classifier-stratified-seed2026.pt
```

For normal installed/package use, you usually do not need to pass the two `--*-checkpoint` options: bundled package checkpoints are auto-detected and enabled. Passing explicit paths is useful when testing a newer checkpoint.

To download official PaddleOCR models on a GitHub runner or other machine before processing:

```bash
ocxkcd download-models --model-root ./models
```

## Local Paddle models

By default PaddleOCR may resolve/download official models. To use an existing local cache, mount or copy it under:

```text
models/paddlex/official_models/PP-OCRv5_server_det
models/paddlex/official_models/PP-OCRv5_server_rec
```

or set:

```bash
OCXKCD_MODEL_ROOT=/path/to/models ocxkcd process image.png -o output
```

## Optional learned refiners

The tool works without learned refiners, but the release package includes the current validated defaults and enables them automatically. If you have compatible replacement checkpoints, use:

```bash
ocxkcd process image.png -o output \
  --crop-refiner-checkpoint /path/to/refiner/best.pt \
  --crop-refiner-combine prefer \
  --fill-prior-checkpoint /path/to/fill-prior/best.pt
```

These options override the bundled defaults.

## Defaults

The CLI defaults are tuned for current xkcd/what-if tests:

- PP-OCRv5 server detector and recognizer
- 2x upscale before OCR
- CPU with `--no-enable-mkldnn` by default, avoiding the known Paddle oneDNN/PIR crash seen in this environment
- text-shaped cleanup masks, not whole OCR rectangles
- interior/local background fill refinement
- dark panel fill handling
- rotated text boxes preserved in `meta.json`

## Output format

`*.meta.json` is compatible with `comic_editor` version 2 metadata:

```json
{
  "version": 2,
  "image": "example.png",
  "cleanBase": true,
  "clearedAreas": [],
  "texts": [
    {
      "id": "text_paddle_0001",
      "key": "caption_1",
      "x": 10,
      "y": 20,
      "width": 80,
      "height": 16,
      "angle": 0,
      "fontSize": 13,
      "fontStyle": "semibold",
      "color": "#000000",
      "align": "left",
      "lineHeight": 1.2
    }
  ]
}
```

`*.text.json` maps caption keys to recognized text:

```json
{
  "caption_1": "HELLO"
}
```

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The default CI tests do not download Paddle models. They validate packaging, manifest generation, and comic_editor export compatibility using deterministic fake OCR input. The Docker publishing workflow downloads PaddleOCR models during image build so the GHCR image is usable without first-run downloads.

## comic_editor integration test

The test suite includes an integration test that calls the real `comic_editor` Skia CLI:

```bash
COMIC_EDITOR_DIR=/path/to/comic_editor pytest -m integration
```

The test uses `COMIC_EDITOR_DIR` when set. Otherwise, it looks for a sibling `comic_editor` checkout next to this repository. On GitHub runners, the CI workflow checks out `comic_editor` and sets `COMIC_EDITOR_DIR` explicitly.
