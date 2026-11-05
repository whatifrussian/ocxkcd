from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


def comic_editor_dir() -> Path | None:
    configured = os.environ.get("COMIC_EDITOR_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).resolve().parents[2] / "comic_editor")
    candidates.append(Path(__file__).resolve().parents[1].parent / "comic_editor")
    for candidate in candidates:
        renderer = candidate / "render_comic_image_skia.mjs"
        if renderer.exists():
            return candidate
    return None


@pytest.mark.integration
@pytest.mark.parametrize("conditioned", [False, True])
def test_generated_export_renders_with_actual_comic_editor_cli(tmp_path: Path, conditioned: bool) -> None:
    from ocxkcd.cli import default_checkpoint
    extra = []
    if conditioned:
        pytest.importorskip("torch")
        extra = ["--cleaner-checkpoint", str(default_checkpoint(None, "conditioned-cleaner-v1.pt"))]
    editor = comic_editor_dir()
    if editor is None:
        pytest.skip("comic_editor checkout not found; set COMIC_EDITOR_DIR to run this integration test")
    if shutil.which("node") is None:
        pytest.skip("node is required for comic_editor CLI integration test")

    source = Path(__file__).parent / "data" / "arrest.png"
    report = {
        "version": 1,
        "records": [
            {
                "key": "arrest",
                "title": "arrest",
                "page": "",
                "source": str(source),
                "width": 200,
                "height": 150,
                "geometry": [],
                "seconds": 0.0,
                "detections": [
                    {
                        "index": 1,
                        "text": "SHOCK",
                        "score": 0.99,
                        "box": {"x": 86, "y": 92, "width": 82, "height": 29, "angle": 0},
                        "poly": [[86, 92], [168, 92], [168, 121], [86, 121]],
                        "bestGeometry": None,
                    }
                ],
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "export"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "ocxkcd._comic_export",
            "--report",
            str(report_path),
            "--output",
            str(output),
            "--min-score",
            "0",
            "--min-chars",
            "1",
            "--keep-single-char",
            "--refine-fill-from-interior",
        ] + extra,
        check=True,
    )

    render_output = output / "arrest" / "arrest.translated.png"
    result = subprocess.run(
        [
            "node",
            str(editor / "render_comic_image_skia.mjs"),
            str(output / "arrest"),
            "arrest",
            "--scale",
            "1",
            "--resample",
            "native",
            "--font-dir",
            str(editor),
            "--output",
            str(render_output),
        ],
        cwd=str(editor),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.fail(f"comic_editor Skia renderer failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    rendered = Image.open(render_output).convert("RGBA")
    original = Image.open(source).convert("RGBA")
    assert rendered.size == original.size
    assert rendered.getbbox() is not None
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["records"][0]["cleaner"]["mode"] == ("conditioned" if conditioned else "legacy")
    empty = tmp_path / "empty.json"
    empty.write_text('{"caption_1": ""}')
    empty_render = tmp_path / "empty.png"
    subprocess.run([
        "node", str(editor / "render_comic_image_skia.mjs"), str(output / "arrest"), "arrest",
        "--scale", "1", "--resample", "native", "--font-dir", str(editor),
        "--text", str(empty), "--output", str(empty_render),
    ], check=True, capture_output=True)
    import numpy as np
    clean = Image.open(output / "arrest" / "arrest.clean.png").convert("RGBA")
    mask = np.array(Image.open(output / "arrest" / "arrest.mask.png"))[..., 3] > 0
    np.testing.assert_array_equal(np.array(clean)[~mask], np.array(original)[~mask])
    for background in ("white", "black"):
        def flatten(im):
            return np.array(Image.alpha_composite(Image.new("RGBA", im.size, background), im)).astype(int)
        assert np.abs(flatten(clean) - flatten(Image.open(empty_render).convert("RGBA"))).max() <= 2
