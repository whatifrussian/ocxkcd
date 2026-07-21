#!/usr/bin/env python3
"""Export full PaddleOCR detections into comic_editor-compatible files.

For each source image this writes a small folder containing:
  <stem>.png         original image copy
  <stem>.clean.png   original with OCR regions filled by estimated background
  <stem>.mask.png    alpha mask for OCR regions
  <stem>.meta.json   comic_editor cleared areas + text boxes
  <stem>.text.json   OCR text keyed by caption_N
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

TEXT_RE = re.compile(r"[A-Za-z0-9]")
SYMBOL_RE = re.compile(r"^[^A-Za-z0-9]+$")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return cleaned.strip("._") or "image"


def compact_text(value: str) -> str:
    return "".join(ch for ch in value.strip() if not ch.isspace())


def accepted_detection(det: dict, args: argparse.Namespace) -> tuple[bool, str]:
    text = (det.get("text") or "").strip()
    score = float(det.get("score") or 0.0)
    box = det.get("box") or {}
    compact = compact_text(text)
    if not text:
        return False, "empty text"
    if "\\" in text:
        return False, "ocr escape/noise text"
    if score < args.min_score:
        return False, "low score"
    if float(box.get("width") or 0) < args.min_width or float(box.get("height") or 0) < args.min_height:
        return False, "tiny box"
    if not TEXT_RE.search(text):
        return False, "no latin/digit text"
    if SYMBOL_RE.match(text):
        return False, "symbol-only"
    if len(compact) < args.min_chars:
        return False, "too few OCR characters"
    if len(compact) == 1 and score < args.min_single_char_score:
        return False, "low-score single character"
    if len(compact) == 1 and not args.keep_single_char:
        return False, "single character"
    if float(box.get("width") or 0) / max(1.0, float(box.get("height") or 0)) < args.min_aspect and len(compact) <= 2:
        return False, "narrow symbol-like detection"
    return True, "accepted"




def box_corners(box: dict) -> list[tuple[float, float]]:
    x = float(box.get("x") or 0)
    y = float(box.get("y") or 0)
    width = float(box.get("width") or 0)
    height = float(box.get("height") or 0)
    angle = math.radians(float(box.get("angle") or 0))
    cos = math.cos(angle)
    sin = math.sin(angle)
    def point(px: float, py: float) -> tuple[float, float]:
        return (x + px * cos - py * sin, y + px * sin + py * cos)
    return [point(0, 0), point(width, 0), point(width, height), point(0, height)]


def crop_bounds(box: dict, size: tuple[int, int], pad: int) -> tuple[int, int, int, int]:
    points = box_corners(box)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return (
        max(0, math.floor(min(xs) - pad)),
        max(0, math.floor(min(ys) - pad)),
        min(size[0], math.ceil(max(xs) + pad)),
        min(size[1], math.ceil(max(ys) + pad)),
    )

def clamp_box(box: dict, width: int, height: int, pad: int = 0) -> dict:
    x0 = max(0, math.floor(float(box["x"]) - pad))
    y0 = max(0, math.floor(float(box["y"]) - pad))
    x1 = min(width, math.ceil(float(box["x"]) + float(box["width"]) + pad))
    y1 = min(height, math.ceil(float(box["y"]) + float(box["height"]) + pad))
    return {"x": x0, "y": y0, "width": max(0, x1 - x0), "height": max(0, y1 - y0)}


def polygon_points(det: dict, width: int, height: int, pad: int = 0) -> list[tuple[int, int]]:
    if det.get("poly"):
        raw = [(float(x), float(y)) for x, y in det["poly"]]
        if pad > 0 and raw:
            cx = sum(x for x, _ in raw) / len(raw)
            cy = sum(y for _, y in raw) / len(raw)
            expanded = []
            for x, y in raw:
                dx = x - cx
                dy = y - cy
                length = max(1.0, math.hypot(dx, dy))
                expanded.append((x + pad * dx / length, y + pad * dy / length))
            raw = expanded
        return [(min(width, max(0, round(x))), min(height, max(0, round(y)))) for x, y in raw]

    b = clamp_box(det["box"], width, height, pad)
    x0, y0 = b["x"], b["y"]
    x1, y1 = x0 + b["width"], y0 + b["height"]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def mask_from_polygon(size: tuple[int, int], points: list[tuple[int, int]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def median_rgba(pixels: np.ndarray) -> tuple[int, int, int, int] | None:
    if pixels.size == 0:
        return None
    med = np.median(pixels.reshape(-1, 4), axis=0)
    return tuple(int(round(v)) for v in med)


def snap_background_color(r: int, g: int, b: int, a: int, args: argparse.Namespace) -> tuple[int, int, int, int]:
    if a >= args.snap_white_min_alpha:
        if min(r, g, b) >= args.snap_white_min_rgb and max(r, g, b) - min(r, g, b) <= args.snap_white_max_delta:
            return 255, 255, 255, 255
    return r, g, b, a


def estimate_background(source: Image.Image, box: dict, inner_mask: Image.Image, args: argparse.Namespace) -> dict:
    rgba = np.asarray(source.convert("RGBA"))
    source_has_transparency = float(np.mean(rgba[:, :, 3] <= args.transparent_alpha)) >= args.source_transparent_ratio
    width, height = source.size
    outer = clamp_box(box, width, height, args.bg_ring)
    inner = clamp_box(box, width, height, max(0, args.padding - 1))

    ring = np.zeros((height, width), dtype=bool)
    ring[outer["y"] : outer["y"] + outer["height"], outer["x"] : outer["x"] + outer["width"]] = True
    ring[inner["y"] : inner["y"] + inner["height"], inner["x"] : inner["x"] + inner["width"]] = False

    inner_region = np.asarray(inner_mask, dtype=np.uint8) > 0
    ring &= ~inner_region

    candidates = rgba[ring]
    # If the image is transparent line art, background samples are alpha-zero.
    if candidates.size == 0:
        candidates = rgba[
            inner["y"] : inner["y"] + inner["height"],
            inner["x"] : inner["x"] + inner["width"],
        ].reshape(-1, 4)
    if candidates.size == 0:
        return {"transparent": True, "fill": (0, 0, 0, 0), "hex": None}

    alpha_zero_ratio = float(np.mean(candidates[:, 3] <= args.transparent_alpha))
    opaque = candidates[candidates[:, 3] > args.transparent_alpha]
    opaque_med = median_rgba(opaque) if opaque.size else None
    opaque_lum = None
    if opaque_med is not None:
        opaque_lum = 0.2126 * opaque_med[0] + 0.7152 * opaque_med[1] + 0.0722 * opaque_med[2]
    # Transparent source art often has black line strokes in the sampling ring.
    # Treat low-alpha background plus dark opaque samples as transparent, not black fill.
    if alpha_zero_ratio >= args.transparent_ratio or (
        source_has_transparency
        and alpha_zero_ratio >= args.transparent_ring_min
        and (opaque_lum is None or opaque_lum <= args.transparent_dark_luma)
    ):
        return {"transparent": True, "fill": (0, 0, 0, 0), "hex": None}

    med = opaque_med or median_rgba(candidates) or (255, 255, 255, 255)
    r, g, b, a = med
    a = max(255 if a > 220 else a, args.transparent_alpha + 1)
    r, g, b, a = snap_background_color(r, g, b, a, args)
    return {"transparent": False, "fill": (r, g, b, a), "hex": f"#{r:02x}{g:02x}{b:02x}"}


def refine_visible_transparent_candidate_background(
    source: Image.Image,
    candidate_mask: Image.Image,
    bg: dict,
    args: argparse.Namespace,
) -> dict:
    """Detect semi-transparent local fills before text-mask extraction.

    Transparent line art often has true transparent background, but some labels are
    semi-transparent black/gray plates with transparent or opaque text on top. If
    we keep the initial background as transparent, the mask extractor sees the
    whole plate as foreground and erases a solid bar. This early pass promotes a
    dominant visible candidate color to fill before text_ink_mask runs.
    """
    if not bool(bg.get("transparent")) or not args.detect_visible_transparent_fill:
        return bg
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    candidate = np.asarray(candidate_mask.convert("L"), dtype=np.uint8) > 0
    candidate_count = int(np.count_nonzero(candidate))
    if candidate_count <= 0:
        return bg
    pixels = rgba[candidate & (rgba[:, :, 3] > args.transparent_alpha)]
    visible_ratio = pixels.shape[0] / max(1, candidate_count)
    if pixels.shape[0] < args.refine_fill_min_pixels or visible_ratio < args.visible_transparent_fill_min_visible_ratio:
        return bg

    alpha = pixels[:, 3].astype(np.int16)
    # Round alpha to reduce anti-alias noise, then choose the dominant visible
    # alpha bucket. This catches the glass water plates at alpha ~= 84 while
    # ignoring sparse opaque text strokes on otherwise transparent art.
    alpha_bucket = np.clip((alpha // args.visible_transparent_fill_alpha_quantize) * args.visible_transparent_fill_alpha_quantize, 0, 255)
    values, counts = np.unique(alpha_bucket, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    winner_ratio = int(counts[int(np.argmax(counts))]) / max(1, pixels.shape[0])
    if winner <= args.transparent_alpha or winner >= args.visible_transparent_fill_max_alpha or winner_ratio < args.visible_transparent_fill_min_alpha_ratio:
        return bg
    bucket_pixels = pixels[np.abs(alpha_bucket - winner) <= args.visible_transparent_fill_alpha_quantize]
    if bucket_pixels.shape[0] < args.refine_fill_min_pixels:
        return bg
    med = np.median(bucket_pixels.reshape(-1, 4), axis=0)
    r, g, b, a = (int(round(v)) for v in med)
    if max(r, g, b) - min(r, g, b) > args.visible_transparent_fill_max_rgb_delta:
        return bg
    return {
        "transparent": False,
        "fill": (r, g, b, a),
        "hex": f"#{r:02x}{g:02x}{b:02x}",
        "visibleTransparentFill": {
            "applied": True,
            "visibleRatio": round(visible_ratio, 4),
            "alphaBucket": winner,
            "alphaRatio": round(winner_ratio, 4),
        },
    }




def refine_background_from_interior(
    source: Image.Image,
    candidate_mask: Image.Image,
    erase_mask: Image.Image,
    bg: dict,
    args: argparse.Namespace,
) -> dict:
    """Refine fill color using local non-erased pixels inside the OCR region.

    The outer ring is good for text on plain white/transparent backgrounds, but it
    fails on labels embedded in dark bars, books, screens, maps, and framed boxes:
    the ring may describe the surrounding art rather than the actual background
    directly behind the letters. Once we have a text-shaped erase mask, the best
    estimate is usually the dominant color inside the OCR polygon excluding the
    text pixels themselves.
    """
    if not args.refine_fill_from_interior:
        return bg
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    candidate = np.asarray(candidate_mask.convert("L"), dtype=np.uint8) > 0
    erased = np.asarray(erase_mask.convert("L"), dtype=np.uint8) > 0
    candidate_area = int(np.count_nonzero(candidate))
    candidate_pixels = rgba[candidate & (rgba[:, :, 3] > args.transparent_alpha)]
    if bool(bg.get("transparent")):
        visible_ratio = candidate_pixels.shape[0] / max(1, candidate_area)
        if visible_ratio < args.refine_fill_transparent_min_visible_ratio:
            return bg
    if args.refine_fill_detect_light_label and candidate_pixels.shape[0] >= args.refine_fill_min_pixels:
        alpha = candidate_pixels[:, 3:4].astype(np.float32) / 255.0
        flat_rgb = candidate_pixels[:, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        bright_neutral = (
            (np.min(flat_rgb, axis=1) >= args.refine_fill_light_label_min_rgb)
            & ((np.max(flat_rgb, axis=1) - np.min(flat_rgb, axis=1)) <= args.refine_fill_light_label_max_delta)
        )
        bright_count = int(np.count_nonzero(bright_neutral))
        bright_ratio = bright_count / max(1, int(candidate_pixels.shape[0]))
        if bright_count >= args.refine_fill_min_pixels and bright_ratio >= args.refine_fill_light_label_min_ratio:
            bright_pixels = candidate_pixels[bright_neutral]
            med = np.median(bright_pixels.reshape(-1, 4), axis=0)
            r, g, b, a = (int(round(v)) for v in med)
            a = max(255 if a > 220 else a, args.transparent_alpha + 1)
            r, g, b, a = snap_background_color(r, g, b, a, args)
            return {
                "transparent": False,
                "fill": (r, g, b, a),
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "refinedFromInterior": True,
                "refinedSampleKind": "light-label",
                "previousFill": tuple(bg.get("fill") or (255, 255, 255, 255)),
                "interiorSamples": bright_count,
                "lightLabelRatio": round(bright_ratio, 4),
            }
    sample = None
    sample_kind = "interior"
    if args.refine_fill_local_ring > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.refine_fill_local_ring * 2 + 1, args.refine_fill_local_ring * 2 + 1))
        local = cv2.dilate(erased.astype(np.uint8), k, iterations=1) > 0
        local_sample = candidate & local & ~erased & (rgba[:, :, 3] > args.transparent_alpha)
        if int(np.count_nonzero(local_sample)) >= args.refine_fill_min_pixels:
            sample = local_sample
            sample_kind = "local-ring"
    if args.refine_fill_exclude_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.refine_fill_exclude_dilate * 2 + 1, args.refine_fill_exclude_dilate * 2 + 1))
        erased = cv2.dilate(erased.astype(np.uint8), k, iterations=1) > 0
    if sample is None:
        sample = candidate & ~erased & (rgba[:, :, 3] > args.transparent_alpha)
    pixels = rgba[sample]
    if pixels.shape[0] < args.refine_fill_min_pixels:
        return bg

    # Quantized dominant color is more robust than a raw median when the OCR box
    # contains a mix of local background plus border/art strokes.
    rgb = pixels[:, :3].astype(np.int16)
    bins = np.clip(rgb // max(1, args.refine_fill_quantize), 0, 255).astype(np.int32)
    packed = bins[:, 0] * 65536 + bins[:, 1] * 256 + bins[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    wb = np.array([(winner // 65536) & 255, (winner // 256) & 255, winner & 255], dtype=np.int16)
    cluster = np.all(np.abs(bins - wb[None, :]) <= 1, axis=1)
    cluster_pixels = pixels[cluster]
    if cluster_pixels.shape[0] < args.refine_fill_min_pixels:
        return bg
    med = np.median(cluster_pixels.reshape(-1, 4), axis=0)
    r, g, b, a = (int(round(v)) for v in med)
    a = max(255 if a > 220 else a, args.transparent_alpha + 1)
    r, g, b, a = snap_background_color(r, g, b, a, args)
    old = tuple(bg.get("fill") or (255, 255, 255, 255))
    old_luma = 0.2126 * old[0] + 0.7152 * old[1] + 0.0722 * old[2]
    new_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    changed = bool(bg.get("transparent")) or abs(old_luma - new_luma) >= args.refine_fill_min_luma_change or np.linalg.norm(np.array(old[:3], dtype=np.int16) - np.array([r, g, b], dtype=np.int16)) >= args.refine_fill_min_color_distance
    if not changed:
        return bg
    return {
        "transparent": False,
        "fill": (r, g, b, a),
        "hex": f"#{r:02x}{g:02x}{b:02x}",
        "refinedFromInterior": True,
        "refinedSampleKind": sample_kind,
        "previousFill": old,
        "interiorSamples": int(cluster_pixels.shape[0]),
    }


def text_ink_mask(source: Image.Image, candidate_mask: Image.Image, bg_fill: tuple[int, int, int, int], bg_transparent: bool, args: argparse.Namespace) -> Image.Image:
    rgba = np.asarray(source.convert("RGBA"))
    candidate = np.asarray(candidate_mask, dtype=np.uint8) > 0
    if not np.any(candidate):
        return Image.new("L", source.size, 0)

    if bg_transparent:
        # Transparent XKCD/what-if line art stores every visible stroke as alpha.
        # Using alpha alone erases nearby drawing lines inside the OCR box. Composite
        # on white and keep only dark visible ink; this is much closer to text shape.
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32)
        flat_rgb = rgb * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
        flat_luma = 0.2126 * flat_rgb[:, :, 0] + 0.7152 * flat_rgb[:, :, 1] + 0.0722 * flat_rgb[:, :, 2]
        ink = candidate & (rgba[:, :, 3] > args.transparent_alpha) & (flat_luma <= args.transparent_ink_luma)
    else:
        rgb = rgba[:, :, :3].astype(np.int16)
        bg = np.array(bg_fill[:3], dtype=np.int16)
        dist = np.linalg.norm(rgb - bg, axis=2)
        alpha_diff = np.abs(rgba[:, :, 3].astype(np.int16) - int(bg_fill[3]))
        base_changed = candidate & (rgba[:, :, 3] > args.transparent_alpha) & (
            (dist >= args.ink_color_distance) | (alpha_diff >= args.ink_alpha_distance)
        )
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        bg_luma = 0.2126 * int(bg_fill[0]) + 0.7152 * int(bg_fill[1]) + 0.0722 * int(bg_fill[2])
        bright = base_changed & (luma >= bg_luma + args.ink_luma_delta)
        dark = base_changed & (luma <= bg_luma - args.ink_luma_delta)
        bright_count = int(np.count_nonzero(bright))
        dark_count = int(np.count_nonzero(dark))
        if bright_count or dark_count:
            if bg_luma >= args.light_background_luma:
                ink = dark if dark_count >= max(1, bright_count) * args.polarity_switch_ratio else bright
            elif bg_luma <= args.dark_background_luma:
                ink = bright if bright_count >= max(1, dark_count) * args.polarity_switch_ratio else dark
            else:
                ink = bright if bright_count >= dark_count else dark
        else:
            ink = base_changed

    b = np.argwhere(candidate)
    y0, x0 = b.min(axis=0)
    y1, x1 = b.max(axis=0) + 1
    crop = ink[y0:y1, x0:x1].astype(np.uint8)
    if crop.size == 0:
        return Image.new("L", source.size, 0)

    # Remove table borders / leader lines that were inside the OCR box, but keep
    # normal letter strokes. This is intentionally conservative: false negatives
    # are easier to repair manually than erased artwork lines.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    filtered = np.zeros_like(crop, dtype=np.uint8)
    region_w = max(1, x1 - x0)
    region_h = max(1, y1 - y0)
    for idx in range(1, n):
        cx, cy, cw, ch, area = map(int, stats[idx])
        if area < args.min_ink_area:
            continue
        horizontal_rule = cw >= region_w * args.rule_width_ratio and ch <= max(args.max_rule_thickness, region_h * args.rule_height_ratio)
        vertical_rule = ch >= region_h * args.rule_width_ratio and cw <= max(args.max_rule_thickness, region_w * args.rule_height_ratio)
        if horizontal_rule or vertical_rule:
            continue
        filtered[labels == idx] = 255

    if args.remove_straight_lines:
        min_line = max(args.hough_min_line_length, int(min(region_w, region_h) * args.hough_min_line_ratio))
        lines = cv2.HoughLinesP(
            filtered,
            rho=1,
            theta=np.pi / 180,
            threshold=args.hough_threshold,
            minLineLength=min_line,
            maxLineGap=args.hough_max_gap,
        )
        if lines is not None:
            removal = np.zeros_like(filtered, dtype=np.uint8)
            for line in np.asarray(lines).reshape(-1, 4):
                x_a, y_a, x_b, y_b = map(int, line)
                length = math.hypot(x_b - x_a, y_b - y_a)
                if length < min_line:
                    continue
                # Preserve normal letter strokes: only remove long strokes that either
                # touch/exit the OCR region border or span a large share of the region.
                touches_edge = (
                    x_a <= 1 or x_b <= 1 or y_a <= 1 or y_b <= 1 or
                    x_a >= region_w - 2 or x_b >= region_w - 2 or
                    y_a >= region_h - 2 or y_b >= region_h - 2
                )
                spans_region = length >= max(region_w, region_h) * args.hough_span_ratio
                if touches_edge or spans_region:
                    cv2.line(removal, (x_a, y_a), (x_b, y_b), 255, args.hough_line_width)
            if np.any(removal):
                filtered[removal > 0] = 0

    if args.ink_close > 0:
        k = np.ones((args.ink_close, args.ink_close), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, k)
    if args.ink_dilate > 0:
        k = np.ones((args.ink_dilate * 2 + 1, args.ink_dilate * 2 + 1), np.uint8)
        filtered = cv2.dilate(filtered, k, iterations=1)

    out = np.zeros(candidate.shape, dtype=np.uint8)
    out[y0:y1, x0:x1] = filtered
    out &= candidate.astype(np.uint8) * 255
    return Image.fromarray(out, mode="L")




def detection_angle_degrees(det: dict) -> float:
    poly = det.get("poly") or []
    if len(poly) >= 2:
        x0, y0 = poly[0]
        x1, y1 = poly[1]
        return math.degrees(math.atan2(float(y1) - float(y0), float(x1) - float(x0)))
    return 0.0


def detection_padding(det: dict, args: argparse.Namespace) -> int:
    if abs(detection_angle_degrees(det)) >= args.rotated_text_min_angle:
        return args.rotated_padding
    return args.padding


def tight_oriented_box(mask: Image.Image, det: dict, image_width: int, image_height: int, args: argparse.Namespace) -> dict:
    alpha = np.asarray(mask.convert("L"), dtype=np.uint8)
    ys, xs = np.nonzero(alpha > 0)
    if len(xs) == 0:
        b = clamp_box(det["box"], image_width, image_height, args.padding)
        return {**b, "angle": 0.0}

    angle = detection_angle_degrees(det)
    theta = math.radians(angle)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)
    # Use pixel centers to avoid systematically shaving off antialiased edge pixels.
    px = xs.astype(np.float64) + 0.5
    py = ys.astype(np.float64) + 0.5
    s = px * ux + py * uy
    t = px * vx + py * vy
    q = max(0.0, min(0.2, float(args.shrink_quantile)))
    box_padding = args.rotated_text_box_padding if abs(angle) >= args.rotated_text_min_angle else args.text_box_padding
    if q > 0 and len(s) >= args.shrink_min_pixels_for_quantile:
        min_s = float(np.quantile(s, q)) - box_padding
        max_s = float(np.quantile(s, 1.0 - q)) + box_padding
        min_t = float(np.quantile(t, q)) - box_padding
        max_t = float(np.quantile(t, 1.0 - q)) + box_padding
    else:
        min_s = float(s.min()) - box_padding
        max_s = float(s.max()) + box_padding
        min_t = float(t.min()) - box_padding
        max_t = float(t.max()) + box_padding

    # Clamp approximately by using axis-aligned bounds after projection. This keeps
    # the metadata sane without forcing broad Paddle boxes back into the result.
    x = min_s * ux + min_t * vx
    y = min_s * uy + min_t * vy
    width = max(1.0, max_s - min_s)
    height = max(1.0, max_t - min_t)
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "angle": round(angle, 2),
    }



def oriented_box_mask(size: tuple[int, int], box: dict) -> Image.Image:
    x = float(box.get("x") or 0)
    y = float(box.get("y") or 0)
    width = float(box.get("width") or 0)
    height = float(box.get("height") or 0)
    angle = math.radians(float(box.get("angle") or 0))
    cos = math.cos(angle)
    sin = math.sin(angle)
    def point(px: float, py: float) -> tuple[int, int]:
        return (round(x + px * cos - py * sin), round(y + px * sin + py * cos))
    return mask_from_polygon(size, [point(0, 0), point(width, 0), point(width, height), point(0, height)])


def shrink_box_along_height(box: dict, ratio: float) -> dict:
    ratio = max(0.1, min(1.0, float(ratio)))
    if ratio >= 0.999:
        return box
    angle = math.radians(float(box.get("angle") or 0))
    vx, vy = -math.sin(angle), math.cos(angle)
    old_height = float(box.get("height") or 0)
    new_height = max(1.0, old_height * ratio)
    offset = (old_height - new_height) / 2.0
    return {
        **box,
        "x": round(float(box.get("x") or 0) + offset * vx, 2),
        "y": round(float(box.get("y") or 0) + offset * vy, 2),
        "height": round(new_height, 2),
    }


def intersect_masks(left: Image.Image, right: Image.Image) -> Image.Image:
    a = np.asarray(left.convert("L"), dtype=np.uint8)
    b = np.asarray(right.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where((a > 0) & (b > 0), 255, 0).astype(np.uint8), mode="L")

def brush_cleanup_mask(ink_mask: Image.Image, text_box: dict, args: argparse.Namespace) -> Image.Image:
    mask = np.asarray(ink_mask.convert("L"), dtype=np.uint8)
    if args.brush_cleanup_radius > 0:
        radius = int(args.brush_cleanup_radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    if args.brush_cleanup_close > 0:
        radius = int(args.brush_cleanup_close)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    expanded_box = dict(text_box)
    extra = float(args.brush_cleanup_clip_padding)
    if extra:
        angle = math.radians(float(expanded_box.get("angle") or 0))
        expanded_box["x"] = float(expanded_box.get("x") or 0) - extra * math.cos(angle) + extra * math.sin(angle)
        expanded_box["y"] = float(expanded_box.get("y") or 0) - extra * math.sin(angle) - extra * math.cos(angle)
        expanded_box["width"] = float(expanded_box.get("width") or 0) + extra * 2
        expanded_box["height"] = float(expanded_box.get("height") or 0) + extra * 2
    clip = np.asarray(oriented_box_mask(ink_mask.size, expanded_box).convert("L"), dtype=np.uint8)
    mask = np.where((mask > 0) & (clip > 0), 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")

def estimate_text_color(source: Image.Image, box: dict, bg_fill: tuple[int, int, int, int]) -> str:
    rgba = np.asarray(source.convert("RGBA"))
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])
    crop = rgba[y : y + h, x : x + w]
    if crop.size == 0:
        return "#000000"
    rgb = crop[:, :, :3].astype(np.int16)
    alpha = crop[:, :, 3]
    bg = np.array(bg_fill[:3], dtype=np.int16)
    dist = np.linalg.norm(rgb - bg, axis=2)
    ink = (alpha > 16) & (dist > 35)
    if not np.any(ink):
        return "#000000"
    med = np.median(rgb[ink], axis=0)
    lum = 0.2126 * med[0] + 0.7152 * med[1] + 0.0722 * med[2]
    return "#ffffff" if lum > 150 else "#000000"


def estimate_font_size(box: dict) -> int:
    return max(8, min(72, round(float(box["height"]) * 0.82)))


def windows_file_url(path: Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        rest = text[7:].replace("\\", "/")
        return f"file:///{drive}:/{rest}"
    return path.resolve().as_uri()


def relative_url(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix())


def make_mask_preview(mask: Image.Image, clean: Image.Image) -> Image.Image:
    width, height = mask.size
    tile = 16
    yy, xx = np.indices((height, width))
    checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    base = np.where(checker[:, :, None] == 0, np.array([238, 242, 247], dtype=np.uint8), np.array([180, 190, 202], dtype=np.uint8))
    clean_rgb = np.asarray(clean.convert("RGB"), dtype=np.uint8)
    alpha = np.asarray(mask.convert("RGBA"), dtype=np.float32)[:, :, 3] / 255.0
    # Show context faintly, then tint actual mask pixels magenta. This makes white
    # text masks visible on white, transparent, black, and colored backgrounds.
    preview = (base.astype(np.float32) * 0.55 + clean_rgb.astype(np.float32) * 0.45).astype(np.uint8)
    tint = np.array([255, 0, 180], dtype=np.float32)
    amount = (alpha * 0.88)[:, :, None]
    preview = (preview.astype(np.float32) * (1.0 - amount) + tint * amount).astype(np.uint8)
    return Image.fromarray(preview, mode="RGB")


def is_meta_region_record(record: dict) -> bool:
    detections = record.get("detections") or []
    return bool(record.get("sourceMeta")) and bool(detections) and all((det.get("source") == "meta-region") for det in detections)


def placeholder_text(det: dict, ordinal: int) -> str:
    text = (det.get("text") or "").strip()
    if text and text != "TEXT":
        return text
    if det.get("source") == "meta-region":
        return f"REGION_{ordinal:03d}"
    return text


RGBA_COLOR_RE = re.compile(
    r"rgba\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)"
)


def repair_preserved_alpha_colors(meta: dict, source: Image.Image, args: argparse.Namespace) -> dict:
    """Keep semi-transparent source colors when preserving reviewed metadata.

    Some early reviewed masks stored semi-transparent black as its white-composited
    appearance, e.g. rgba(171,171,171,1). That renders correctly only on white.
    For transparent source art we need the actual source color/alpha instead:
    rgba(0,0,0,0.329).
    """

    def repair_area(area: dict) -> None:
        color = area.get("color")
        if not color or bool(area.get("transparent")):
            return
        match = RGBA_COLOR_RE.fullmatch(str(color).strip())
        if not match:
            return
        r, g, b = (int(round(float(match.group(i)))) for i in (1, 2, 3))
        color_alpha = float(match.group(4))
        if color_alpha < 0.99 or max(r, g, b) - min(r, g, b) > 4 or not (24 <= r <= 245):
            return
        x0 = max(0, int(math.floor(float(area.get("x") or 0))))
        y0 = max(0, int(math.floor(float(area.get("y") or 0))))
        x1 = min(source.size[0], int(math.ceil(float(area.get("x") or 0) + float(area.get("width") or 0))))
        y1 = min(source.size[1], int(math.ceil(float(area.get("y") or 0) + float(area.get("height") or 0))))
        if x1 <= x0 or y1 <= y0:
            return
        pixels = np.asarray(source.crop((x0, y0, x1, y1)).convert("RGBA"), dtype=np.uint8).reshape(-1, 4)
        visible = pixels[pixels[:, 3] > args.transparent_alpha]
        if visible.shape[0] < args.refine_fill_min_pixels:
            return
        med = np.median(visible, axis=0)
        sr, sg, sb, sa = (int(round(v)) for v in med)
        if max(sr, sg, sb) <= 8 and args.transparent_alpha < sa < 245:
            alpha = sa / 255.0
            area["color"] = f"rgba(0, 0, 0, {alpha:.4f})"
            area["opacity"] = alpha
            area["alphaRepairedFromSource"] = True

    for area in meta.get("clearedAreas") or []:
        repair_area(area)
    proposal = meta.get("proposal")
    if isinstance(proposal, dict):
        for area in proposal.get("shapes") or []:
            repair_area(area)
    return meta


def maybe_dark_panel_fill(source: Image.Image, text_box: dict, bg: dict, args: argparse.Namespace) -> dict:
    if not args.dark_panel_fill:
        return bg
    x0, y0, x1, y1 = crop_bounds(text_box, source.size, 0)
    if x1 <= x0 or y1 <= y0:
        return bg
    rgba = np.asarray(source.crop((x0, y0, x1, y1)).convert("RGBA"), dtype=np.uint8).reshape(-1, 4)
    if rgba.shape[0] < args.refine_fill_min_pixels:
        return bg
    alpha = rgba[:, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :3].astype(np.float32)
    flat = rgb * alpha[:, None] + 255.0 * (1.0 - alpha[:, None])
    luma = 0.2126 * flat[:, 0] + 0.7152 * flat[:, 1] + 0.0722 * flat[:, 2]
    opaque_ratio = float(np.mean(rgba[:, 3] >= args.dark_panel_min_alpha))
    dark_ratio = float(np.mean((rgba[:, 3] >= args.dark_panel_min_alpha) & (luma <= args.dark_panel_max_luma)))
    if opaque_ratio >= args.dark_panel_min_opaque_ratio and dark_ratio >= args.dark_panel_min_dark_ratio:
        return {
            "transparent": False,
            "fill": (0, 0, 0, 255),
            "hex": "#000000",
            "darkPanelFill": {
                "applied": True,
                "opaqueRatio": round(opaque_ratio, 4),
                "darkRatio": round(dark_ratio, 4),
            },
        }
    return bg


def preserve_reviewed_artifacts(record: dict, source: Image.Image, output_root: Path, args: argparse.Namespace) -> dict | None:
    if args.meta_region_artifacts != "preserve":
        return None
    if not is_meta_region_record(record):
        return None
    source_meta = Path(record["sourceMeta"])
    if not source_meta.exists():
        return None

    review_dir = source_meta.parent
    source_path = Path(record["source"])
    key = record.get("key") or source_path.name
    key_parts = [safe_name(part) for part in Path(key).parts]
    if key_parts and key_parts[-1].lower().endswith(".png"):
        key_parts[-1] = Path(key_parts[-1]).stem
    folder = output_root.joinpath(*key_parts)
    folder.mkdir(parents=True, exist_ok=True)

    stem = safe_name(source_path.stem)
    original_out = folder / f"{stem}.png"
    clean_out = folder / f"{stem}.clean.png"
    mask_out = folder / f"{stem}.mask.png"
    mask_preview_out = folder / f"{stem}.mask.preview.png"
    meta_out = folder / f"{stem}.meta.json"
    text_out = folder / f"{stem}.text.json"
    raw_out = folder / f"{stem}.paddle.json"
    box_preview_out = folder / f"{stem}.boxes.preview.png"

    shutil.copy2(source_path, original_out)
    reviewed_clean = review_dir / f"{source_path.stem}.clean.png"
    reviewed_mask = review_dir / f"{source_path.stem}.mask.png"
    reviewed_text = review_dir / f"{source_path.stem}.text.json"

    if reviewed_clean.exists():
        shutil.copy2(reviewed_clean, clean_out)
    else:
        source.save(clean_out)
    if reviewed_mask.exists():
        shutil.copy2(reviewed_mask, mask_out)
    else:
        Image.new("RGBA", source.size, (255, 255, 255, 0)).save(mask_out)
    reviewed_meta = json.loads(source_meta.read_text(encoding="utf-8"))
    reviewed_meta = repair_preserved_alpha_colors(reviewed_meta, source, args)
    meta_out.write_text(json.dumps(reviewed_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if reviewed_text.exists():
        shutil.copy2(reviewed_text, text_out)
    else:
        text_out.write_text("{}\n", encoding="utf-8")

    with Image.open(clean_out) as clean_file, Image.open(mask_out) as mask_file:
        make_mask_preview(mask_file.convert("RGBA"), clean_file.convert("RGBA")).save(mask_preview_out)

    exported = []
    for ordinal, det in enumerate(record.get("detections", []), start=1):
        exported.append(
            {
                "index": det.get("index"),
                "text": placeholder_text(det, ordinal),
                "textSource": "manual-region",
                "score": det.get("score"),
                "box": det.get("box"),
                "transparent": None,
                "color": None,
                "textColor": None,
                "maskPixels": None,
                "artifactMode": "preserved-reviewed",
            }
        )
    raw_out.write_text(
        json.dumps(
            {
                "source": record.get("source"),
                "key": key,
                "artifactMode": "preserved-reviewed",
                "sourceMeta": str(source_meta),
                "accepted": exported,
                "rejected": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    make_text_box_preview(source, exported).save(box_preview_out)

    return {
        "key": key,
        "folder": str(folder),
        "folderUrl": windows_file_url(folder),
        "source": str(source_path),
        "image": relative_url(original_out, output_root),
        "clean": relative_url(clean_out, output_root),
        "mask": relative_url(mask_out, output_root),
        "maskPreview": relative_url(mask_preview_out, output_root),
        "meta": relative_url(meta_out, output_root),
        "text": relative_url(text_out, output_root),
        "boxPreview": relative_url(box_preview_out, output_root),
        "accepted": len(exported),
        "rejected": 0,
        "detections": exported,
        "artifactMode": "preserved-reviewed",
    }




def load_unet_prior(args):
    if not args.unet_checkpoint:
        return None
    import torch
    ml_root = Path(__file__).resolve().parents[1] / "ml"
    if str(ml_root) not in sys.path:
        sys.path.insert(0, str(ml_root))
    from ocxkcd._ml_data import image_tensor, letterbox
    from ocxkcd._ml_model import SmallUNet

    use_cuda = torch.cuda.is_available() and args.unet_device != "cpu"
    if args.unet_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for U-Net prior but is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = torch.load(args.unet_checkpoint, map_location=device)
    model = SmallUNet(base_channels=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return {
        "torch": torch,
        "model": model,
        "device": device,
        "image_size": int(checkpoint["image_size"]),
        "image_tensor": image_tensor,
        "letterbox": letterbox,
        "checkpoint": str(args.unet_checkpoint),
    }


def predict_unet_prior(source: Image.Image, prior: dict | None, threshold: float) -> Image.Image | None:
    if prior is None:
        return None
    torch = prior["torch"]
    prepared, _, (left, top, width, height) = prior["letterbox"](source, None, prior["image_size"])
    with torch.no_grad():
        probability = torch.sigmoid(prior["model"](prior["image_tensor"](prepared).unsqueeze(0).to(prior["device"])))[0, 0]
    content = probability[top : top + height, left : left + width].detach().cpu()
    values = (content >= threshold).to(torch.uint8).mul(255).contiguous()
    predicted = Image.frombytes("L", (width, height), bytes(values.flatten().tolist()))
    return predicted.resize(source.size, Image.Resampling.NEAREST)


def combine_with_unet(cleanup_mask: Image.Image, unet_mask: Image.Image | None, text_box: dict, args: argparse.Namespace) -> tuple[Image.Image, str]:
    if unet_mask is None or args.unet_combine == "off":
        return cleanup_mask, "off"
    cleanup = np.asarray(cleanup_mask.convert("L"), dtype=np.uint8) > 0
    prior = np.asarray(unet_mask.convert("L"), dtype=np.uint8) > 0
    clip = np.asarray(oriented_box_mask(cleanup_mask.size, text_box).convert("L"), dtype=np.uint8) > 0
    if args.unet_prior_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.unet_prior_dilate * 2 + 1, args.unet_prior_dilate * 2 + 1))
        prior = cv2.dilate(prior.astype(np.uint8), k, iterations=1) > 0
    if args.unet_combine == "intersect":
        combined = cleanup & prior & clip
    elif args.unet_combine == "union":
        combined = (cleanup | prior) & clip
    elif args.unet_combine == "prefer":
        prior_pixels = int(np.count_nonzero(prior & clip))
        cleanup_pixels = int(np.count_nonzero(cleanup & clip))
        if cleanup_pixels and args.unet_prefer_min_ratio <= prior_pixels / cleanup_pixels <= args.unet_prefer_max_ratio:
            combined = prior & clip
        else:
            combined = cleanup & clip
    else:
        raise ValueError(f"unknown --unet-combine: {args.unet_combine}")
    return Image.fromarray(np.where(combined, 255, 0).astype(np.uint8), mode="L"), args.unet_combine



def load_crop_refiner(args):
    if not args.crop_refiner_checkpoint:
        return None
    import torch
    ml_root = Path(__file__).resolve().parents[1] / "ml"
    if str(ml_root) not in sys.path:
        sys.path.insert(0, str(ml_root))
    from ocxkcd._ml_data import image_tensor, letterbox
    from ocxkcd._ml_model import SmallUNet

    use_cuda = torch.cuda.is_available() and args.crop_refiner_device != "cpu"
    if args.crop_refiner_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for crop refiner but is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = torch.load(args.crop_refiner_checkpoint, map_location=device)
    model = SmallUNet(base_channels=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return {
        "torch": torch,
        "model": model,
        "device": device,
        "image_size": int(checkpoint["image_size"]),
        "image_tensor": image_tensor,
        "letterbox": letterbox,
        "checkpoint": str(args.crop_refiner_checkpoint),
    }


def predict_crop_refiner(source: Image.Image, crop_box: tuple[int, int, int, int], refiner: dict | None, threshold: float) -> Image.Image | None:
    if refiner is None:
        return None
    x0, y0, x1, y1 = crop_box
    if x1 <= x0 or y1 <= y0:
        return Image.new("L", source.size, 0)
    crop = source.crop(crop_box).convert("RGBA")
    torch = refiner["torch"]
    prepared, _, (left, top, width, height) = refiner["letterbox"](crop, None, refiner["image_size"])
    with torch.no_grad():
        probability = torch.sigmoid(refiner["model"](refiner["image_tensor"](prepared).unsqueeze(0).to(refiner["device"])))[0, 0]
    content = probability[top : top + height, left : left + width].detach().cpu()
    values = (content >= threshold).to(torch.uint8).mul(255).contiguous()
    predicted_crop = Image.frombytes("L", (width, height), bytes(values.flatten().tolist()))
    predicted_crop = predicted_crop.resize(crop.size, Image.Resampling.NEAREST)
    out = Image.new("L", source.size, 0)
    out.paste(predicted_crop, (x0, y0))
    return out


def combine_with_crop_refiner(cleanup_mask: Image.Image, refiner_mask: Image.Image | None, text_box: dict, args: argparse.Namespace) -> tuple[Image.Image, str]:
    if refiner_mask is None or args.crop_refiner_combine == "off":
        return cleanup_mask, "off"
    cleanup = np.asarray(cleanup_mask.convert("L"), dtype=np.uint8) > 0
    learned = np.asarray(refiner_mask.convert("L"), dtype=np.uint8) > 0
    clip = np.asarray(oriented_box_mask(cleanup_mask.size, text_box).convert("L"), dtype=np.uint8) > 0
    if args.crop_refiner_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.crop_refiner_dilate * 2 + 1, args.crop_refiner_dilate * 2 + 1))
        learned = cv2.dilate(learned.astype(np.uint8), k, iterations=1) > 0
    cleanup_pixels = int(np.count_nonzero(cleanup & clip))
    learned_pixels = int(np.count_nonzero(learned & clip))
    if args.crop_refiner_combine == "intersect":
        combined = cleanup & learned & clip
    elif args.crop_refiner_combine == "union":
        combined = (cleanup | learned) & clip
    elif args.crop_refiner_combine == "replace":
        combined = learned & clip
    elif args.crop_refiner_combine == "prefer":
        ratio = learned_pixels / max(1, cleanup_pixels)
        if learned_pixels >= args.min_detection_mask_pixels and args.crop_refiner_prefer_min_ratio <= ratio <= args.crop_refiner_prefer_max_ratio:
            combined = learned & clip
        else:
            combined = cleanup & clip
    else:
        raise ValueError(f"unknown --crop-refiner-combine: {args.crop_refiner_combine}")
    return Image.fromarray(np.where(combined, 255, 0).astype(np.uint8), mode="L"), args.crop_refiner_combine


class FillPriorNet:
    """Lazy wrapper for the crop-level fill classifier.

    The checkpoint was trained with a tiny 5-channel CNN: flattened RGBA crop plus
    final erase-mask alpha. We define the module inside the loader so importing
    this exporter stays cheap unless --fill-prior-checkpoint is used.
    """


def load_fill_prior(args):
    if not args.fill_prior_checkpoint:
        return None
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, 4)
        def forward(self, x):
            return self.head(self.net(x).flatten(1))

    use_cuda = torch.cuda.is_available() and args.fill_prior_device != "cpu"
    if args.fill_prior_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for fill prior but is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = torch.load(args.fill_prior_checkpoint, map_location=device)
    labels = list(checkpoint.get("labels") or ["transparent", "white", "black", "other"])
    model = Net().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return {
        "torch": torch,
        "model": model,
        "device": device,
        "labels": labels,
        "image_size": int(checkpoint.get("image_size") or 96),
        "checkpoint": str(args.fill_prior_checkpoint),
    }


def _fill_prior_tensor(source_crop: Image.Image, mask_crop: Image.Image, size: int, torch):
    image = source_crop.convert("RGBA").resize((size, size), Image.Resampling.BILINEAR)
    mask = mask_crop.convert("L").resize((size, size), Image.Resampling.NEAREST)
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flattened = Image.alpha_composite(white, image)
    rgba = torch.tensor(bytearray(flattened.tobytes()), dtype=torch.uint8).view(size, size, 4).permute(2, 0, 1).float().div_(255)
    mask_tensor = torch.tensor(bytearray(mask.tobytes()), dtype=torch.uint8).view(size, size).unsqueeze(0).float().div_(255)
    return torch.cat([rgba, mask_tensor], dim=0).unsqueeze(0)


def predict_fill_prior(source: Image.Image, cleanup_mask: Image.Image, text_box: dict, prior: dict | None, args: argparse.Namespace) -> dict | None:
    if prior is None:
        return None
    mask_array = np.asarray(cleanup_mask.convert("L"), dtype=np.uint8) > 0
    ys, xs = np.nonzero(mask_array)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - args.fill_prior_padding)
    y0 = max(0, int(ys.min()) - args.fill_prior_padding)
    x1 = min(source.size[0], int(xs.max()) + 1 + args.fill_prior_padding)
    y1 = min(source.size[1], int(ys.max()) + 1 + args.fill_prior_padding)
    crop = (x0, y0, x1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    mask_crop = cleanup_mask.crop(crop).convert("L")
    if not np.any(np.asarray(mask_crop, dtype=np.uint8) > 0):
        return None
    torch = prior["torch"]
    tensor = _fill_prior_tensor(source.crop(crop), mask_crop, prior["image_size"], torch).to(prior["device"])
    with torch.no_grad():
        probabilities = torch.softmax(prior["model"](tensor), dim=1)[0].detach().cpu().tolist()
    best_index = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))
    label = prior["labels"][best_index] if best_index < len(prior["labels"]) else str(best_index)
    return {
        "label": label,
        "confidence": float(probabilities[best_index]),
        "probabilities": {prior["labels"][i] if i < len(prior["labels"]) else str(i): float(value) for i, value in enumerate(probabilities)},
        "crop": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
    }


def apply_fill_prior(bg: dict, prediction: dict | None, args: argparse.Namespace) -> dict:
    if prediction is None:
        return bg
    label = prediction["label"]
    confidence = float(prediction["confidence"])
    min_confidence = args.fill_prior_min_confidence
    if label == "white" and args.fill_prior_white_min_confidence is not None:
        min_confidence = args.fill_prior_white_min_confidence
    elif label == "black" and args.fill_prior_black_min_confidence is not None:
        min_confidence = args.fill_prior_black_min_confidence
    elif label == "transparent" and args.fill_prior_transparent_min_confidence is not None:
        min_confidence = args.fill_prior_transparent_min_confidence
    if confidence < min_confidence:
        return {**bg, "fillPrior": {**prediction, "applied": False, "reason": "low confidence"}}
    if label == "transparent":
        if not args.fill_prior_allow_transparent:
            return {**bg, "fillPrior": {**prediction, "applied": False, "reason": "transparent disabled"}}
        return {"transparent": True, "fill": (0, 0, 0, 0), "hex": None, "fillPrior": {**prediction, "applied": True}}
    if label == "white":
        current_fill = bg.get("fill") or (0, 0, 0, 0)
        current_rgb = [int(current_fill[0]), int(current_fill[1]), int(current_fill[2])]
        current_luma = 0.2126 * int(current_fill[0]) + 0.7152 * int(current_fill[1]) + 0.0722 * int(current_fill[2])
        if args.fill_prior_white_guard_light and not bool(bg.get("transparent")) and current_luma < args.light_background_luma and confidence < args.fill_prior_min_confidence:
            return {**bg, "fillPrior": {**prediction, "applied": False, "reason": "white low confidence would override dark heuristic"}}
        current_neutral_white = min(current_rgb) >= args.fill_prior_white_neutral_min_rgb and max(current_rgb) - min(current_rgb) <= args.fill_prior_white_neutral_max_delta
        if args.fill_prior_white_guard_neutral and not bool(bg.get("transparent")) and not current_neutral_white and confidence < args.fill_prior_white_strong_confidence:
            return {**bg, "fillPrior": {**prediction, "applied": False, "reason": "white low confidence would override local nonwhite fill"}}
        return {"transparent": False, "fill": (255, 255, 255, 255), "hex": "#ffffff", "fillPrior": {**prediction, "applied": True}}
    if label == "black":
        return {"transparent": False, "fill": (0, 0, 0, 255), "hex": "#000000", "fillPrior": {**prediction, "applied": True}}
    if label == "other" and args.fill_prior_apply_other:
        return {**bg, "fillPrior": {**prediction, "applied": False, "reason": "other keeps heuristic color"}}
    return {**bg, "fillPrior": {**prediction, "applied": False, "reason": "unsupported label"}}

def export_record(record: dict, output_root: Path, args: argparse.Namespace, unet_prior: dict | None = None, crop_refiner: dict | None = None, fill_prior: dict | None = None) -> dict:
    source_path = Path(record["source"])
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGBA")
    width, height = source.size

    preserved = preserve_reviewed_artifacts(record, source, output_root, args)
    if preserved is not None:
        return preserved

    key = record.get("key") or source_path.name
    key_parts = [safe_name(part) for part in Path(key).parts]
    if key_parts and key_parts[-1].lower().endswith(".png"):
        key_parts[-1] = Path(key_parts[-1]).stem
    folder = output_root.joinpath(*key_parts)
    folder.mkdir(parents=True, exist_ok=True)

    stem = safe_name(source_path.stem)
    original_out = folder / f"{stem}.png"
    clean_out = folder / f"{stem}.clean.png"
    mask_out = folder / f"{stem}.mask.png"
    mask_preview_out = folder / f"{stem}.mask.preview.png"
    meta_out = folder / f"{stem}.meta.json"
    text_out = folder / f"{stem}.text.json"
    raw_out = folder / f"{stem}.paddle.json"
    box_preview_out = folder / f"{stem}.boxes.preview.png"

    shutil.copy2(source_path, original_out)

    unet_mask = predict_unet_prior(source, unet_prior, args.unet_threshold)

    clean = source.copy()
    region_mask = Image.new("RGBA", source.size, (255, 255, 255, 0))
    region_draw = ImageDraw.Draw(region_mask)
    meta = {"version": 2, "image": f"{stem}.png", "cleanBase": True, "clearedAreas": [], "texts": []}
    text_values: dict[str, str] = {}
    exported = []
    rejected = []

    for det in record.get("detections", []):
        ok, reason = accepted_detection(det, args)
        if not ok:
            rejected.append({"index": det.get("index"), "text": det.get("text"), "score": det.get("score"), "reason": reason})
            continue
        pad = detection_padding(det, args)
        padded_box = clamp_box(det["box"], width, height, pad)
        if padded_box["width"] <= 0 or padded_box["height"] <= 0:
            continue
        points = polygon_points(det, width, height, pad)
        candidate_mask = mask_from_polygon(source.size, points)
        bg = estimate_background(source, padded_box, candidate_mask, args)
        bg = refine_visible_transparent_candidate_background(source, candidate_mask, bg, args)
        fill = tuple(bg["fill"])
        ink_mask = text_ink_mask(source, candidate_mask, fill, bool(bg["transparent"]), args)
        mask_pixels = int(np.count_nonzero(np.asarray(ink_mask, dtype=np.uint8)))
        if mask_pixels < args.min_detection_mask_pixels:
            rejected.append({"index": det.get("index"), "text": det.get("text"), "score": det.get("score"), "reason": "no text-shaped mask pixels"})
            continue

        text_box = tight_oriented_box(ink_mask, det, width, height, args)
        if bg.get("visibleTransparentFill") and args.visible_transparent_text_box_height_ratio < 1.0:
            text_box = shrink_box_along_height(text_box, args.visible_transparent_text_box_height_ratio)
        if args.clip_mask_to_tight_box:
            tight_mask = oriented_box_mask(source.size, text_box)
            ink_mask = intersect_masks(ink_mask, tight_mask)
            mask_pixels = int(np.count_nonzero(np.asarray(ink_mask, dtype=np.uint8)))
            if mask_pixels < args.min_detection_mask_pixels:
                rejected.append({"index": det.get("index"), "text": det.get("text"), "score": det.get("score"), "reason": "empty after tight box shrink"})
                continue
            text_box = tight_oriented_box(ink_mask, det, width, height, args)

        cleanup_mask = brush_cleanup_mask(ink_mask, text_box, args)
        cleanup_mask, unet_mode = combine_with_unet(cleanup_mask, unet_mask, text_box, args)
        crop_refiner_mode = "off"
        if crop_refiner is not None and args.crop_refiner_combine != "off":
            refiner_crop = crop_bounds(text_box, source.size, args.crop_refiner_padding)
            learned_mask = predict_crop_refiner(source, refiner_crop, crop_refiner, args.crop_refiner_threshold)
            cleanup_mask, crop_refiner_mode = combine_with_crop_refiner(cleanup_mask, learned_mask, text_box, args)
        cleanup_pixels = int(np.count_nonzero(np.asarray(cleanup_mask, dtype=np.uint8)))
        if cleanup_pixels < args.min_detection_mask_pixels:
            rejected.append({"index": det.get("index"), "text": det.get("text"), "score": det.get("score"), "reason": "empty after brush cleanup"})
            continue

        bg = refine_background_from_interior(source, candidate_mask, cleanup_mask, bg, args)
        bg = maybe_dark_panel_fill(source, text_box, bg, args)
        fill_prior_prediction = predict_fill_prior(source, cleanup_mask, text_box, fill_prior, args)
        bg = apply_fill_prior(bg, fill_prior_prediction, args)
        fill = tuple(bg["fill"])

        # Clean with a flexible brush-like mask. Rectangles remain text placement
        # boxes, not erase instructions.
        fill_layer = Image.new("RGBA", source.size, fill)
        clean.paste(fill_layer, (0, 0), cleanup_mask)
        region_mask.paste(Image.new("RGBA", source.size, (255, 255, 255, 255)), (0, 0), cleanup_mask)
        mask_pixels = cleanup_pixels

        key_name = f"caption_{len(exported) + 1}"
        text_value = placeholder_text(det, len(exported) + 1)
        text_values[key_name] = "" if det.get("source") == "meta-region" and text_value.startswith("REGION_") else text_value
        font_size = estimate_font_size(text_box)
        text_color = estimate_text_color(source, padded_box, fill)
        if args.emit_rect_clears:
            meta["clearedAreas"].append(
                {
                    "type": "rect",
                    "x": text_box["x"],
                    "y": text_box["y"],
                    "width": text_box["width"],
                    "height": text_box["height"],
                    "transparent": bool(bg["transparent"]),
                    "color": bg["hex"],
                }
            )
        meta["texts"].append(
            {
                "id": f"text_paddle_{len(exported) + 1:04d}",
                "key": key_name,
                "x": text_box["x"],
                "y": text_box["y"],
                "width": text_box["width"],
                "height": text_box["height"],
                "angle": text_box["angle"],
                "fontSize": font_size,
                "fontStyle": "semibold",
                "color": text_color,
                "align": "left",
                "lineHeight": 1.2,
            }
        )
        exported.append(
            {
                "index": det.get("index"),
                "text": text_value,
                "textSource": "manual-region" if det.get("source") == "meta-region" else "ocr",
                "score": det.get("score"),
                "box": text_box,
                "transparent": bool(bg["transparent"]),
                "color": bg["hex"],
                "fillRgba": list(fill),
                "textColor": text_color,
                "maskPixels": mask_pixels,
                "unetMode": unet_mode if args.unet_checkpoint else "off",
                "cropRefinerMode": crop_refiner_mode if args.crop_refiner_checkpoint else "off",
                "darkPanelFill": bg.get("darkPanelFill"),
                "fillPrior": bg.get("fillPrior"),
            }
        )

    clean.save(clean_out)
    region_mask.save(mask_out)
    make_mask_preview(region_mask, clean).save(mask_preview_out)
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_out.write_text(json.dumps(text_values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_out.write_text(json.dumps({"source": record.get("source"), "key": key, "accepted": exported, "rejected": rejected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_text_box_preview(source, exported).save(box_preview_out)

    return {
        "key": key,
        "folder": str(folder),
        "folderUrl": windows_file_url(folder),
        "source": str(source_path),
        "image": relative_url(original_out, output_root),
        "clean": relative_url(clean_out, output_root),
        "mask": relative_url(mask_out, output_root),
        "maskPreview": relative_url(mask_preview_out, output_root),
        "meta": relative_url(meta_out, output_root),
        "text": relative_url(text_out, output_root),
        "boxPreview": relative_url(box_preview_out, output_root),
        "accepted": len(exported),
        "rejected": len(rejected),
        "detections": exported,
    }




def make_text_box_preview(source: Image.Image, detections: list[dict]) -> Image.Image:
    preview = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source.convert("RGBA"))
    draw = ImageDraw.Draw(preview)
    for item in detections:
        box = item.get("box") or {}
        x = float(box.get("x") or 0)
        y = float(box.get("y") or 0)
        width = float(box.get("width") or 0)
        height = float(box.get("height") or 0)
        angle = math.radians(float(box.get("angle") or 0))
        cos = math.cos(angle)
        sin = math.sin(angle)
        def point(px: float, py: float) -> tuple[float, float]:
            return (x + px * cos - py * sin, y + px * sin + py * cos)
        corners = [point(0, 0), point(width, 0), point(width, height), point(0, height)]
        draw.line(corners + [corners[0]], fill=(0, 210, 255, 255), width=2)
        if abs(float(box.get("angle") or 0)) > 0.25:
            # Draw orientation baseline in orange so rotated boxes are obvious.
            draw.line([corners[0], corners[1]], fill=(255, 132, 0, 255), width=3)
        label = str(item.get("index") or "")
        if label:
            lx, ly = corners[0]
            draw.rectangle((lx, ly - 12, lx + max(10, len(label) * 7), ly), fill=(0, 210, 255, 220))
            draw.text((lx + 2, ly - 12), label, fill=(0, 0, 0, 255))
    return preview.convert("RGB")


def write_index(output_root: Path, records: list[dict], summary: dict) -> None:
    cards = []
    for rec in records:
        rows = "".join(
            f"<tr><td>{d['index']}</td><td>{html.escape(d['text'])}</td><td>{html.escape(d.get('textSource') or '')}</td><td>{float(d.get('score') or 0):.3f}</td>"
            f"<td>{html.escape(str(d['transparent']))}</td><td>{html.escape(str(d['color']))}<small>{html.escape(str(d.get('fillRgba') or ''))}</small></td></tr>"
            for d in rec["detections"]
        )
        cards.append(
            f"<section><h2>{html.escape(rec['key'])}</h2>"
            f"<p><a href='{html.escape(rec['folderUrl'])}'>open folder</a> · accepted {rec['accepted']} · rejected {rec['rejected']}</p>"
            f"<div class='imgs'><figure><img src='{rec['image']}'><figcaption>original</figcaption></figure>"
            f"<figure><img src='{rec['clean']}'><figcaption>clean, color matched</figcaption></figure>"
            f"<figure><img src='{rec['maskPreview']}'><figcaption>mask preview: magenta=text-shaped erase</figcaption></figure>"
            f"<figure><img src='{rec['boxPreview']}'><figcaption>text boxes: cyan=box, orange=rotation baseline</figcaption></figure></div>"
            f"<p><a href='{rec['meta']}'>meta.json</a> · <a href='{rec['text']}'>text.json</a></p>"
            f"<table><thead><tr><th>#</th><th>text / region</th><th>source</th><th>score</th><th>transparent</th><th>color</th></tr></thead><tbody>{rows}</tbody></table>"
            f"</section>"
        )
    document = f"""<!doctype html><meta charset='utf-8'><title>Paddle to comic_editor export</title>
<style>
body{{font:15px system-ui;margin:24px;background:#eef1f4;color:#1f2933}}section{{background:white;border:1px solid #ccd4dc;border-radius:9px;padding:14px;margin:14px 0 22px}}.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}.card{{background:white;border-radius:8px;padding:12px 16px;border:1px solid #ccd4dc}}.imgs{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;align-items:start}}figure{{margin:0}}img{{max-width:100%;border:1px solid #cbd5e1;background-color:#cbd5e1;background-image:linear-gradient(45deg,#eef2f7 25%,transparent 25%),linear-gradient(-45deg,#eef2f7 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#eef2f7 75%),linear-gradient(-45deg,transparent 75%,#eef2f7 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}}figcaption{{font-size:12px;color:#64748b}}table{{width:100%;border-collapse:collapse;margin-top:10px}}td,th{{border:1px solid #d1d5db;padding:5px 7px;text-align:left;vertical-align:top}}
</style><h1>Paddle to comic_editor export</h1>
<div class='summary'><div class='card'>images: {summary['images']}</div><div class='card'>accepted detections: {summary['accepted']}</div><div class='card'>rejected detections: {summary['rejected']}</div></div>
<p>Open any generated folder in the browser editor. Each folder contains original, clean, mask, meta, text, and raw Paddle files.</p>
{''.join(cards)}"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("report.json"))
    parser.add_argument("--output", type=Path, default=Path("comic_editor_export"))
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=6.0)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--min-single-char-score", type=float, default=0.5)
    parser.add_argument("--keep-single-char", action="store_true")
    parser.add_argument("--min-aspect", type=float, default=0.2)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--text-box-padding", type=float, default=1.0)
    parser.add_argument("--rotated-text-min-angle", type=float, default=12.0)
    parser.add_argument("--rotated-padding", type=int, default=0)
    parser.add_argument("--rotated-text-box-padding", type=float, default=0.35)
    parser.add_argument("--shrink-quantile", type=float, default=0.01)
    parser.add_argument("--shrink-min-pixels-for-quantile", type=int, default=80)
    parser.add_argument("--clip-mask-to-tight-box", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--brush-cleanup-radius", type=int, default=1)
    parser.add_argument("--brush-cleanup-close", type=int, default=0)
    parser.add_argument("--brush-cleanup-clip-padding", type=float, default=0.5)
    parser.add_argument("--bg-ring", type=int, default=12)
    parser.add_argument("--transparent-alpha", type=int, default=24)
    parser.add_argument("--transparent-ratio", type=float, default=0.55)
    parser.add_argument("--source-transparent-ratio", type=float, default=0.02)
    parser.add_argument("--transparent-ring-min", type=float, default=0.08)
    parser.add_argument("--transparent-dark-luma", type=float, default=80.0)
    parser.add_argument("--detect-visible-transparent-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible-transparent-fill-min-visible-ratio", type=float, default=0.72)
    parser.add_argument("--visible-transparent-fill-alpha-quantize", type=int, default=8)
    parser.add_argument("--visible-transparent-fill-min-alpha-ratio", type=float, default=0.45)
    parser.add_argument("--visible-transparent-fill-max-alpha", type=int, default=245)
    parser.add_argument("--visible-transparent-fill-max-rgb-delta", type=int, default=16)
    parser.add_argument("--visible-transparent-text-box-height-ratio", type=float, default=0.58)
    parser.add_argument("--snap-white-min-rgb", type=int, default=240)
    parser.add_argument("--snap-white-max-delta", type=int, default=18)
    parser.add_argument("--snap-white-min-alpha", type=int, default=220)
    parser.add_argument("--transparent-ink-luma", type=float, default=225.0)
    parser.add_argument("--ink-color-distance", type=float, default=34.0)
    parser.add_argument("--ink-luma-delta", type=float, default=52.0)
    parser.add_argument("--light-background-luma", type=float, default=170.0)
    parser.add_argument("--dark-background-luma", type=float, default=135.0)
    parser.add_argument("--polarity-switch-ratio", type=float, default=0.35)
    parser.add_argument("--ink-alpha-distance", type=float, default=40.0)
    parser.add_argument("--min-ink-area", type=int, default=3)
    parser.add_argument("--ink-dilate", type=int, default=0)
    parser.add_argument("--min-detection-mask-pixels", type=int, default=8)
    parser.add_argument("--ink-close", type=int, default=0)
    parser.add_argument("--remove-straight-lines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hough-threshold", type=int, default=8)
    parser.add_argument("--hough-min-line-length", type=int, default=20)
    parser.add_argument("--hough-min-line-ratio", type=float, default=0.55)
    parser.add_argument("--hough-max-gap", type=int, default=3)
    parser.add_argument("--hough-line-width", type=int, default=3)
    parser.add_argument("--hough-span-ratio", type=float, default=0.55)
    parser.add_argument("--rule-width-ratio", type=float, default=0.82)
    parser.add_argument("--rule-height-ratio", type=float, default=0.12)
    parser.add_argument("--max-rule-thickness", type=int, default=3)
    parser.add_argument("--unet-checkpoint", type=Path)
    parser.add_argument("--unet-threshold", type=float, default=0.5)
    parser.add_argument("--unet-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--unet-combine", default="off", choices=("off", "intersect", "union", "prefer"))
    parser.add_argument("--unet-prior-dilate", type=int, default=1)
    parser.add_argument("--unet-prefer-min-ratio", type=float, default=0.35)
    parser.add_argument("--unet-prefer-max-ratio", type=float, default=1.8)
    parser.add_argument("--crop-refiner-checkpoint", type=Path, help="crop-level text/artwork refiner checkpoint trained from review exports")
    parser.add_argument("--crop-refiner-threshold", type=float, default=0.5)
    parser.add_argument("--crop-refiner-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--crop-refiner-combine", default="off", choices=("off", "intersect", "union", "prefer", "replace"))
    parser.add_argument("--crop-refiner-padding", type=int, default=26)
    parser.add_argument("--crop-refiner-dilate", type=int, default=0)
    parser.add_argument("--crop-refiner-prefer-min-ratio", type=float, default=0.25)
    parser.add_argument("--crop-refiner-prefer-max-ratio", type=float, default=2.2)
    parser.add_argument("--refine-fill-from-interior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refine-fill-min-pixels", type=int, default=12)
    parser.add_argument("--refine-fill-local-ring", type=int, default=3)
    parser.add_argument("--refine-fill-detect-light-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-fill-light-label-min-rgb", type=int, default=188)
    parser.add_argument("--refine-fill-light-label-max-delta", type=int, default=36)
    parser.add_argument("--refine-fill-light-label-min-ratio", type=float, default=0.18)
    parser.add_argument("--refine-fill-transparent-min-visible-ratio", type=float, default=0.45)
    parser.add_argument("--refine-fill-exclude-dilate", type=int, default=2)
    parser.add_argument("--refine-fill-quantize", type=int, default=16)
    parser.add_argument("--refine-fill-min-luma-change", type=float, default=18.0)
    parser.add_argument("--refine-fill-min-color-distance", type=float, default=28.0)
    parser.add_argument("--fill-prior-checkpoint", type=Path, help="optional crop-level fill classifier checkpoint")
    parser.add_argument("--fill-prior-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--fill-prior-padding", type=int, default=10)
    parser.add_argument("--fill-prior-min-confidence", type=float, default=0.90)
    parser.add_argument("--fill-prior-white-min-confidence", type=float)
    parser.add_argument("--fill-prior-black-min-confidence", type=float)
    parser.add_argument("--fill-prior-transparent-min-confidence", type=float)
    parser.add_argument("--fill-prior-white-guard-light", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-prior-white-guard-neutral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-prior-white-strong-confidence", type=float, default=0.75)
    parser.add_argument("--fill-prior-white-neutral-min-rgb", type=int, default=235)
    parser.add_argument("--fill-prior-white-neutral-max-delta", type=int, default=24)
    parser.add_argument("--fill-prior-allow-transparent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-prior-apply-other", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dark-panel-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dark-panel-min-alpha", type=int, default=220)
    parser.add_argument("--dark-panel-max-luma", type=float, default=36.0)
    parser.add_argument("--dark-panel-min-opaque-ratio", type=float, default=0.62)
    parser.add_argument("--dark-panel-min-dark-ratio", type=float, default=0.60)
    parser.add_argument("--meta-region-artifacts", default="regenerate", choices=("regenerate", "preserve"), help="For synthetic meta-region records, either run the detector pipeline or preserve existing reviewed clean/mask/meta artifacts.")
    parser.add_argument("--emit-rect-clears", action="store_true", help="Also emit old rectangle clear areas. Off by default because clean/mask are text-shaped.")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    unet_prior = load_unet_prior(args)
    crop_refiner = load_crop_refiner(args)
    fill_prior = load_fill_prior(args)
    records = [export_record(record, args.output, args, unet_prior, crop_refiner, fill_prior) for record in report.get("records", [])]
    summary = {
        "images": len(records),
        "accepted": sum(record["accepted"] for record in records),
        "rejected": sum(record["rejected"] for record in records),
    }
    (args.output / "manifest.json").write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_index(args.output, records, summary)
    print(json.dumps({"output": str(args.output), "index": windows_file_url(args.output / "index.html"), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
