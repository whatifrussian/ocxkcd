#!/usr/bin/env bash
set -euo pipefail

MODE=copy
SOURCE_ROOT=
TARGET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

usage() {
  cat <<'EOF'
Usage: scripts/prepare-models.sh [--copy|--symlink] [--source-root PATH] [--target-root PATH]

Copies or symlinks model artifacts from an existing training/export checkout
into ocxkcd's runtime model layout.

Required source layout:
  SOURCE_ROOT/_models/paddlex/official_models/PP-OCRv5_server_det
  SOURCE_ROOT/_models/paddlex/official_models/PP-OCRv5_server_rec
  SOURCE_ROOT/_text_artwork_training_v1/runs/refiner-smoke-seed1337/best.pt
  SOURCE_ROOT/_fill_training_v1/runs/fill-classifier-stratified-seed2026/best.pt

Default target:
  ./models
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy)
      MODE=copy
      shift
      ;;
    --symlink)
      MODE=symlink
      shift
      ;;
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --target-root)
      TARGET_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "--source-root is required" >&2
  usage >&2
  exit 2
fi

copy_or_link() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "Missing source artifact: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst"
  if [[ "$MODE" == "symlink" ]]; then
    ln -s "$src" "$dst"
  else
    cp -a "$src" "$dst"
  fi
  echo "$MODE $src -> $dst"
}

copy_or_link \
  "$SOURCE_ROOT/_models/paddlex/official_models/PP-OCRv5_server_det" \
  "$TARGET_ROOT/paddlex/official_models/PP-OCRv5_server_det"

copy_or_link \
  "$SOURCE_ROOT/_models/paddlex/official_models/PP-OCRv5_server_rec" \
  "$TARGET_ROOT/paddlex/official_models/PP-OCRv5_server_rec"

copy_or_link \
  "$SOURCE_ROOT/_text_artwork_training_v1/runs/refiner-smoke-seed1337/best.pt" \
  "$TARGET_ROOT/ocxkcd/refiner-smoke-seed1337.pt"

copy_or_link \
  "$SOURCE_ROOT/_fill_training_v1/runs/fill-classifier-stratified-seed2026/best.pt" \
  "$TARGET_ROOT/ocxkcd/fill-classifier-stratified-seed2026.pt"

cat <<EOF

Done.

Use with:
  ocxkcd process image.png -o output \\
    --model-root "$TARGET_ROOT" \\
    --crop-refiner-checkpoint "$TARGET_ROOT/ocxkcd/refiner-smoke-seed1337.pt" \\
    --crop-refiner-combine prefer \\
    --fill-prior-checkpoint "$TARGET_ROOT/ocxkcd/fill-classifier-stratified-seed2026.pt"
EOF
