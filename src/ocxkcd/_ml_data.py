from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def standardized_rgba(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    red, green, blue = rgb.split()
    return Image.merge("RGBA", (red, green, blue, alpha))


def letterbox(
    image: Image.Image,
    mask: Image.Image | None,
    size: int,
) -> tuple[Image.Image, Image.Image | None, tuple[int, int, int, int]]:
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized_image = standardized_rgba(image).resize((width, height), Image.Resampling.BILINEAR)
    image_canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    left = (size - width) // 2
    top = (size - height) // 2
    image_canvas.paste(resized_image, (left, top))
    mask_canvas = None
    if mask is not None:
        resized_mask = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_canvas = Image.new("L", (size, size), 0)
        mask_canvas.paste(resized_mask, (left, top))
    return image_canvas, mask_canvas, (left, top, width, height)


def image_tensor(image: Image.Image) -> torch.Tensor:
    values = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    return values.view(image.height, image.width, 4).permute(2, 0, 1).float().div_(255)


def mask_tensor(mask: Image.Image) -> torch.Tensor:
    values = torch.tensor(bytearray(mask.tobytes()), dtype=torch.uint8)
    return values.view(mask.height, mask.width).unsqueeze(0).float().div_(255)


class MaskDataset(Dataset):
    def __init__(self, manifest: Path, split: str, size: int, augment: bool = False, return_weights: bool = False) -> None:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.root = manifest.parent
        self.entries = [entry for entry in data["entries"] if entry["split"] == split]
        self.size = size
        self.augment = augment
        self.return_weights = return_weights

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        with Image.open(self.root / entry["image"]) as image_file:
            image = image_file.convert("RGBA")
        with Image.open(self.root / entry["mask"]) as mask_file:
            mask = mask_file.convert("RGBA").getchannel("A")
        image, mask, _ = letterbox(image, mask, self.size)
        assert mask is not None
        weight = None
        if self.return_weights:
            if entry.get("weight"):
                with Image.open(self.root / entry["weight"]) as weight_file:
                    source_weight = weight_file.convert("L")
            else:
                source_weight = Image.new("L", (int(entry["width"]), int(entry["height"])), 255)
            scale = min(self.size / source_weight.width, self.size / source_weight.height)
            width = max(1, round(source_weight.width * scale))
            height = max(1, round(source_weight.height * scale))
            resized_weight = source_weight.resize((width, height), Image.Resampling.BILINEAR)
            weight = Image.new("L", (self.size, self.size), 0)
            weight.paste(resized_weight, ((self.size - width) // 2, (self.size - height) // 2))
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if weight is not None:
                weight = weight.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.return_weights:
            assert weight is not None
            return image_tensor(image), mask_tensor(mask), mask_tensor(weight), float(entry.get("sampleWeight", 1.0))
        return image_tensor(image), mask_tensor(mask)
