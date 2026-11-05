"""Proposal-conditioned pixel cleaner. Coverage is independent of fill RGBA."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing),
            nn.SiLU(),
        )


class ConditionedCleaner(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.a = Block(5, width)
        self.b = Block(width, width * 2)
        self.c = Block(width * 2, width * 4)
        self.d = Block(width * 4, width * 8)
        self.u = Block(width * 12, width * 4)
        self.v = Block(width * 6, width * 2)
        self.w = Block(width * 3, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(F.avg_pool2d(c, 2))

        def up(x, skip):
            return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)

        return self.out(self.w(up(self.v(up(self.u(up(d, c)), b)), a)))


def input_array(rgba, proposal):
    """Visible RGB over white plus ORIGINAL alpha plus proposed coverage."""
    rgba = rgba.astype(np.float32) / 255
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1 - alpha)
    return np.concatenate([rgb, alpha, proposal[..., None].astype(np.float32) / 255], 2)


def predict(model, rgba, proposal, device, tile=256, overlap=64):
    """Native-resolution overlapping tiles; no whole-page downsampling."""
    h, w = proposal.shape
    values = input_array(rgba, proposal)
    accum = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)

    def starts(n):
        return sorted(set(list(range(0, max(1, n - tile + 1), tile - overlap)) + [max(0, n - tile)]))

    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 0.05).astype(np.float32)
    with torch.inference_mode():
        for y in starts(h):
            for x in starts(w):
                hh, ww = min(tile, h - y), min(tile, w - x)
                block = np.ones((tile, tile, 5), np.float32)
                block[..., 3:] = 0
                block[:hh, :ww] = values[y : y + hh, x : x + ww]
                tensor = torch.from_numpy(block.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                pred = model(tensor).sigmoid()[0, 0, :hh, :ww].cpu().numpy()
                accum[y : y + hh, x : x + ww] += pred * window[:hh, :ww]
                weights[y : y + hh, x : x + ww] += window[:hh, :ww]
    return accum / np.maximum(weights, 1e-8)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "conditioned-cleaner-v1":
        raise ValueError("Not a conditioned-cleaner checkpoint")
    model = ConditionedCleaner(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def bounded_mask(probability, proposal, threshold, radius=8):
    """Only refine near supplied OCR proposals; never invent remote erase areas."""
    support = cv2.dilate((proposal > 0).astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    return (probability >= threshold) & support


def predict_detections(model, source, proposal, detections, device):
    """Use detected-line crops with 26 pixels of surrounding context.
    Whole-page tiles change foreground density and normalization statistics.
    Overlapping crops are blended; untouched areas retain the automatic mask.
    """
    h, w = proposal.shape
    total = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    for detection in detections:
        b = detection["box"]
        x, y = float(b["x"]), float(b["y"])
        bw, bh = float(b["width"]), float(b["height"])
        angle = math.radians(float(b.get("angle", 0)))
        c, s = math.cos(angle), math.sin(angle)
        pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in [(0, 0), (bw, 0), (bw, bh), (0, bh)]]
        x0 = max(0, math.floor(min(p[0] for p in pts) - 26))
        x1 = min(w, math.ceil(max(p[0] for p in pts) + 26))
        y0 = max(0, math.floor(min(p[1] for p in pts) - 26))
        y1 = min(h, math.ceil(max(p[1] for p in pts) + 26))
        if x1 <= x0 or y1 <= y0:
            continue
        p = predict(model, source[y0:y1, x0:x1], proposal[y0:y1, x0:x1], device)
        window = np.full(p.shape, 0.1, np.float32)
        local = np.int32([[px - x0, py - y0] for px, py in pts])
        cv2.fillPoly(window, [local], 1)
        total[y0:y1, x0:x1] += p * window
        weight[y0:y1, x0:x1] += window
    return np.where(weight > 0, total / np.maximum(weight, 1e-8), (proposal > 0).astype(np.float32))


def apply_coverage(source, old_clean, proposal, selected):
    """Copy baseline RGBA fills; new pixels use the closest existing fill.

    This deliberately isolates mask quality from background-estimation changes.
    It preserves partial transparency and never uses a manual clean target.
    """
    result = source.copy()
    if not np.any(proposal):
        return result
    _, labels = cv2.distanceTransformWithLabels((proposal == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    palette = old_clean[proposal > 0]
    fills = palette[np.maximum(labels - 1, 0)]
    result[selected] = fills[selected]
    return result


def refine(source, clean, proposal, detections, cleaner):
    model, checkpoint, device = cleaner
    probability = predict_detections(model, source, proposal, detections, device)
    selected = bounded_mask(probability, proposal, checkpoint["threshold"])
    return apply_coverage(source, clean, proposal, selected), selected
