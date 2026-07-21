from __future__ import annotations

import json
from pathlib import Path

from ocxkcd.cli import collect_images, default_checkpoint, write_manifest


def test_collect_images_and_write_manifest(tmp_path: Path) -> None:
    image = tmp_path / "hello world.png"
    image.write_bytes(b"not an actual image for manifest-only test")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("ignore", encoding="utf-8")

    images = collect_images([tmp_path], recursive=False)
    assert images == [image.resolve()]

    manifest = write_manifest(images, tmp_path / "work")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["items"] == []
    assert data["comics"][0]["key"] == "hello_world"
    assert data["comics"][0]["source"] == str(image.resolve())


def test_collect_images_can_exclude_translated_suffix(tmp_path: Path) -> None:
    original = tmp_path / "panel.png"
    translated = tmp_path / "panel_ru.png"
    original.write_bytes(b"x")
    translated.write_bytes(b"x")

    images = collect_images([tmp_path], recursive=False, exclude_globs=["*_ru.*"])

    assert images == [original.resolve()]


def test_bundled_checkpoints_are_available() -> None:
    refiner = default_checkpoint(None, "refiner-smoke-seed1337.pt")
    fill = default_checkpoint(None, "fill-classifier-stratified-seed2026.pt")
    assert refiner is not None and refiner.exists()
    assert fill is not None and fill.exists()
