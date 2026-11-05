#!/usr/bin/env python3
"""Package a selected cleaner checkpoint without machine-specific training paths."""

import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
if checkpoint.get("architecture") != "conditioned-cleaner-v1":
    raise ValueError("Unsupported cleaner architecture")
checkpoint = {
    key: checkpoint[key]
    for key in (
        "architecture",
        "width",
        "model",
        "threshold",
        "epoch",
        "seed",
        "validation",
        "datasetSha256",
        "groups",
    )
}
root = Path(__file__).resolve().parents[1]
for directory in (root / "models/ocxkcd", root / "src/ocxkcd/models/ocxkcd"):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, directory / "conditioned-cleaner-v1.pt")
