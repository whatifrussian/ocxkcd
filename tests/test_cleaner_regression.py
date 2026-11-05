"""Optional pixel-exact check against an approved cleaner comparison report.

Set OCXKCD_CLEANER_REFERENCE to its manifest.json. Assets stay outside the
package; this check uses automatic baseline proposals, never human masks.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_approved_full_comic_outputs():
    manifest_path = os.environ.get("OCXKCD_CLEANER_REFERENCE")
    if not manifest_path:
        pytest.skip("Set OCXKCD_CLEANER_REFERENCE for the full-comic regression set")
    torch = pytest.importorskip("torch")
    from ocxkcd._conditioned_cleaner import load_checkpoint, refine
    from ocxkcd.cli import default_checkpoint

    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(default_checkpoint(None, "conditioned-cleaner-v1.pt"))
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    records = json.loads(manifest_path.read_text())["records"]
    assert len(records) >= 40
    for record in records:
        baseline_path = Path(record["baselineManifest"])
        if not baseline_path.is_absolute():
            baseline_path = next(
                (parent / baseline_path for parent in (root, *root.parents) if (parent / baseline_path).is_file()), baseline_path
            )
        baseline = next(r for r in json.loads(baseline_path.read_text())["records"] if r["key"] == record["key"])

        def rgba(path):
            return np.array(Image.open(path).convert("RGBA"))

        proposal = rgba(baseline_path.parent / baseline["mask"])[..., 3]
        clean, selected = refine(
            rgba(root / record["original"]),
            rgba(root / record["baseline"]),
            proposal,
            baseline["detections"],
            (model, checkpoint, "cpu"),
        )
        np.testing.assert_array_equal(clean, rgba(root / record["candidate"]), err_msg=record["key"])
        expected_mask = rgba(Path(record["folder"]) / (record["stem"] + ".mask.png"))[..., 3] > 0
        np.testing.assert_array_equal(selected, expected_mask, err_msg=record["key"])
