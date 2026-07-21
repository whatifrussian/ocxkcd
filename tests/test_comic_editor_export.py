from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_exporter_writes_comic_editor_compatible_files(tmp_path: Path) -> None:
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
        ],
        check=True,
    )

    folder = output / "arrest"
    assert (folder / "arrest.png").exists()
    assert (folder / "arrest.clean.png").exists()
    assert (folder / "arrest.mask.png").exists()
    assert (folder / "arrest.meta.json").exists()
    assert (folder / "arrest.text.json").exists()

    meta = json.loads((folder / "arrest.meta.json").read_text(encoding="utf-8"))
    text = json.loads((folder / "arrest.text.json").read_text(encoding="utf-8"))
    assert meta["version"] == 2
    assert meta["image"] == "arrest.png"
    assert meta["cleanBase"] is True
    assert meta["texts"][0]["key"] == "caption_1"
    assert text == {"caption_1": "SHOCK"}

    mask = Image.open(folder / "arrest.mask.png").convert("RGBA")
    assert mask.getchannel("A").getbbox() is not None

