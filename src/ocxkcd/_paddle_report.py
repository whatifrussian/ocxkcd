#!/usr/bin/env python3
"""Run full PaddleOCR detection+recognition and compare it to geometry boxes."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw


def default_model_root() -> Path:
    return Path(os.environ.get("OCXKCD_MODEL_ROOT", "models")).resolve()


LOCAL_MODEL_DIRS = {
    "PP-OCRv5_server_det": Path("paddlex/official_models/PP-OCRv5_server_det"),
    "PP-OCRv5_server_rec": Path("paddlex/official_models/PP-OCRv5_server_rec"),
}


def local_model_dir(model_name: str) -> Path | None:
    relative = LOCAL_MODEL_DIRS.get(model_name)
    if relative is None:
        return None
    candidate = default_model_root() / relative
    return candidate if (candidate / "inference.yml").exists() else None


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def flatten_image(path: Path, output: Path, scale: int) -> tuple[int, int]:
    with Image.open(path) as source_file:
        source = source_file.convert("RGBA")
    flattened = Image.new("RGBA", source.size, "white")
    flattened.alpha_composite(source)
    image = flattened.convert("RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    image.save(output)
    return source.size


def as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def scale_point(point, scale: int) -> list[float]:
    return [round(as_float(point[0]) / scale, 3), round(as_float(point[1]) / scale, 3)]


def scale_box(box, scale: int) -> dict:
    values = [as_float(value) / scale for value in box]
    x0, y0, x1, y1 = values
    return {
        "x": round(min(x0, x1), 3),
        "y": round(min(y0, y1), 3),
        "width": round(abs(x1 - x0), 3),
        "height": round(abs(y1 - y0), 3),
    }


def poly_box(poly: list[list[float]]) -> dict:
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    return {
        "x": round(min(xs), 3),
        "y": round(min(ys), 3),
        "width": round(max(xs) - min(xs), 3),
        "height": round(max(ys) - min(ys), 3),
    }


def box_iou(first: dict, second: dict) -> float:
    ax0, ay0 = first["x"], first["y"]
    ax1, ay1 = ax0 + first["width"], ay0 + first["height"]
    bx0, by0 = second["x"], second["y"]
    bx1, by1 = bx0 + second["width"], by0 + second["height"]
    overlap_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = overlap_width * overlap_height
    union = first["width"] * first["height"] + second["width"] * second["height"] - intersection
    return intersection / max(1.0, union)


def box_overlap_fraction(first: dict, second: dict) -> float:
    ax0, ay0 = first["x"], first["y"]
    ax1, ay1 = ax0 + first["width"], ay0 + first["height"]
    bx0, by0 = second["x"], second["y"]
    bx1, by1 = bx0 + second["width"], by0 + second["height"]
    overlap_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    return (overlap_width * overlap_height) / max(1.0, min(first["width"] * first["height"], second["width"] * second["height"]))


def collect_comics(manifests: list[Path]) -> list[dict]:
    comics = []
    seen = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        geometry_by_key = {}
        for item in manifest["items"]:
            geometry_by_key.setdefault(item["imageKey"], []).append(
                {
                    "id": item["id"],
                    "box": item["box"],
                    "sourceRect": item["sourceRect"],
                    "sourceQuad": item["sourceQuad"],
                    "image": item["image"],
                }
            )
        for comic in manifest["comics"]:
            unique_key = f"{manifest_path.parent}:{comic['key']}"
            if unique_key in seen:
                continue
            seen.add(unique_key)
            comics.append(
                {
                    "manifest": str(manifest_path),
                    "manifestRoot": str(manifest_path.parent),
                    "key": comic["key"],
                    "title": comic["title"],
                    "page": comic.get("page", ""),
                    "source": comic["source"],
                    "geometry": geometry_by_key.get(comic["key"], []),
                }
            )
    return comics


def run_full_ocr(args, comics: list[dict]) -> list[dict]:
    from paddleocr import PaddleOCR

    det_dir = local_model_dir(args.det_model)
    rec_dir = local_model_dir(args.rec_model)
    ocr_kwargs = dict(
        text_detection_model_name=args.det_model,
        text_recognition_model_name=args.rec_model,
        text_recognition_batch_size=args.batch_size,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=args.det_limit_side_len,
        text_det_limit_type=args.det_limit_type,
        text_det_thresh=args.det_thresh,
        text_det_box_thresh=args.det_box_thresh,
        text_det_unclip_ratio=args.det_unclip_ratio,
        device=args.device,
        enable_mkldnn=args.enable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    if det_dir is not None:
        ocr_kwargs["text_detection_model_dir"] = str(det_dir)
    if rec_dir is not None:
        ocr_kwargs["text_recognition_model_dir"] = str(rec_dir)
    ocr = PaddleOCR(**ocr_kwargs)
    results = []
    with tempfile.TemporaryDirectory(prefix="neuro-wattif-full-paddleocr-") as temporary_name:
        temporary = Path(temporary_name)
        for index, comic in enumerate(comics, 1):
            prepared = temporary / f"{index:05d}.png"
            width, height = flatten_image(Path(comic["source"]), prepared, args.upscale)
            started = time.perf_counter()
            prediction = ocr.predict(input=str(prepared))[0].json["res"]
            seconds = time.perf_counter() - started
            detections = []
            texts = prediction.get("rec_texts", [])
            scores = prediction.get("rec_scores", [])
            polys = prediction.get("rec_polys", [])
            boxes = prediction.get("rec_boxes", [])
            for number, text in enumerate(texts):
                poly = []
                if number < len(polys):
                    poly = [scale_point(point, args.upscale) for point in polys[number]]
                box = scale_box(boxes[number], args.upscale) if number < len(boxes) else poly_box(poly)
                best = None
                for geometry in comic["geometry"]:
                    overlap = box_overlap_fraction(box, geometry["box"])
                    iou = box_iou(box, geometry["box"])
                    if best is None or (overlap, iou) > (best["overlap"], best["iou"]):
                        best = {"id": geometry["id"], "overlap": round(overlap, 4), "iou": round(iou, 4)}
                detections.append(
                    {
                        "index": number + 1,
                        "text": str(text).strip(),
                        "score": round(float(scores[number]), 6) if number < len(scores) else None,
                        "box": box,
                        "poly": poly,
                        "bestGeometry": best,
                    }
                )
            results.append(
                {
                    **comic,
                    "width": width,
                    "height": height,
                    "seconds": round(seconds, 4),
                    "detections": detections,
                    "rawTextDetectionParams": prediction.get("text_det_params", {}),
                }
            )
            print(json.dumps({"image": comic["key"], "detections": len(detections), "seconds": round(seconds, 3)}), flush=True)
    return results


def draw_overlay(record: dict, output: Path) -> None:
    with Image.open(record["source"]) as source_file:
        source = source_file.convert("RGBA")
    flattened = Image.new("RGBA", source.size, "white")
    flattened.alpha_composite(source)
    overlay = flattened.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    for item in record["geometry"]:
        quad = [(point["x"], point["y"]) for point in item["sourceQuad"]]
        draw.line([*quad, quad[0]], fill=(0, 180, 220, 255), width=2)
    for detection in record["detections"]:
        poly = [(point[0], point[1]) for point in detection["poly"]]
        score = detection.get("score") or 0.0
        overlap = (detection.get("bestGeometry") or {}).get("overlap", 0.0)
        if overlap >= 0.45:
            color = (20, 190, 70, 255)
        elif detection["text"] and score >= 0.65:
            color = (255, 120, 0, 255)
        else:
            color = (230, 60, 60, 255)
        if len(poly) >= 3:
            draw.line([*poly, poly[0]], fill=color, width=2)
            label_x, label_y = poly[0]
        else:
            box = detection["box"]
            xy = [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]]
            draw.rectangle(xy, outline=color, width=2)
            label_x, label_y = box["x"], box["y"]
        draw.text((label_x + 2, label_y + 2), str(detection["index"]), fill=(180, 0, 0, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)


def write_report(output: Path, records: list[dict], args) -> None:
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    sections = []
    total_detections = 0
    high_conf_new = 0
    matched = 0
    for record in records:
        overlay_name = f"{safe_filename(record['manifestRoot'] + '_' + record['key'])}.png"
        draw_overlay(record, overlays / overlay_name)
        rows = []
        total_detections += len(record["detections"])
        for detection in record["detections"]:
            best = detection.get("bestGeometry") or {}
            if best.get("overlap", 0) >= 0.45:
                matched += 1
            elif detection["text"] and (detection.get("score") or 0) >= 0.65:
                high_conf_new += 1
            rows.append(
                "<tr>"
                f"<td>{detection['index']}</td>"
                f"<td>{html.escape(detection['text'] or '∅')}</td>"
                f"<td>{(detection.get('score') or 0):.3f}</td>"
                f"<td>{html.escape(best.get('id', '—'))}<small>overlap {best.get('overlap', 0):.2f}; IoU {best.get('iou', 0):.2f}</small></td>"
                f"<td><small>x={detection['box']['x']:.1f} y={detection['box']['y']:.1f} w={detection['box']['width']:.1f} h={detection['box']['height']:.1f}</small></td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2><a href='{html.escape(record.get('page', ''), quote=True)}'>{html.escape(record['title'])}</a></h2>"
            f"<p>{len(record['geometry'])} geometry boxes; {len(record['detections'])} PaddleOCR detections; {record['seconds']:.2f}s.</p>"
            f"<img class='overlay' src='overlays/{html.escape(overlay_name)}' loading='lazy'>"
            f"<details open><summary>Detections</summary><table><thead><tr><th>#</th><th>Text</th><th>Score</th><th>Best geometry match</th><th>Paddle box</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details></section>"
        )
    payload = {
        "version": 1,
        "provenance": "full PaddleOCR detection+recognition on white-flattened source images",
        "detModel": args.det_model,
        "recModel": args.rec_model,
        "device": args.device,
        "upscale": args.upscale,
        "params": {
            "detLimitSideLen": args.det_limit_side_len,
            "detLimitType": args.det_limit_type,
            "detThresh": args.det_thresh,
            "detBoxThresh": args.det_box_thresh,
            "detUnclipRatio": args.det_unclip_ratio,
            "enableMkldnn": args.enable_mkldnn,
            "cpuThreads": args.cpu_threads,
        },
        "summary": {
            "images": len(records),
            "geometryBoxes": sum(len(record["geometry"]) for record in records),
            "paddleDetections": total_detections,
            "matchedPaddleDetections": matched,
            "highConfidencePotentialNewDetections": high_conf_new,
        },
        "records": records,
    }
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document = f"""<!doctype html><meta charset='utf-8'><title>Full PaddleOCR proposal report</title>
<style>
body{{font:15px system-ui;margin:24px;background:#eef1f4;color:#1f2933}}a{{color:#075985}}section{{background:white;border:1px solid #ccd4dc;border-radius:9px;padding:14px;margin:14px 0 22px}}.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}.card{{background:white;border-radius:8px;padding:12px 16px;border:1px solid #ccd4dc}}.overlay{{display:block;max-width:100%;max-height:720px;margin:0 auto 14px;background:white;border:1px solid #ccc}}table{{width:100%;border-collapse:collapse;table-layout:fixed}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th:first-child,td:first-child{{width:48px}}th:nth-child(3),td:nth-child(3){{width:70px}}small{{display:block;color:#64748b;margin-top:4px}}.legend span{{display:inline-block;margin-right:18px}}.cyan{{color:#009ec3}}.green{{color:#14a44d}}.orange{{color:#d97706}}.red{{color:#dc2626}}
</style><h1>Full PaddleOCR proposal report</h1>
<p>Full PaddleOCR detection + recognition on white-flattened source images. Cyan = our v20 geometry boxes. Green = Paddle detection overlapping our geometry. Orange = high-confidence Paddle text not matched to our geometry. Red = low-confidence or empty Paddle detection.</p>
<div class='summary'><div class='card'><b>{len(records)} images</b><br>{payload['summary']['geometryBoxes']} geometry boxes<br>{total_detections} Paddle detections</div><div class='card'><b>{matched}</b> Paddle detections matched geometry<br><b>{high_conf_new}</b> high-confidence potential new detections</div><div class='card'>det={html.escape(args.det_model)}<br>rec={html.escape(args.rec_model)}<br>upscale={args.upscale}×<br>mkldnn={html.escape(str(args.enable_mkldnn))}</div></div>
{''.join(sections)}"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-model", default="PP-OCRv5_server_det")
    parser.add_argument("--rec-model", default="PP-OCRv5_server_rec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--det-limit-side-len", type=int, default=960)
    parser.add_argument("--det-limit-type", default="max")
    parser.add_argument("--det-thresh", type=float, default=0.2)
    parser.add_argument("--det-box-thresh", type=float, default=0.3)
    parser.add_argument("--det-unclip-ratio", type=float, default=2.0)
    parser.add_argument("--enable-mkldnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = [path.resolve() for path in args.manifest]
    comics = collect_comics(manifests)
    if args.limit is not None:
        comics = comics[: args.limit]
    records = run_full_ocr(args, comics)
    write_report(output, records, args)
    print(json.dumps({"output": str(output), "images": len(records)}, ensure_ascii=False))
    print(output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
