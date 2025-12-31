import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import imagehash
from ultralytics import YOLOWorld
import json


def _iter_images(src: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in src.iterdir() if p.suffix.lower() in exts]


def _resolution_ok(img_path: Path, min_w: int, min_h: int) -> bool:
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        return w >= min_w and h >= min_h
    except Exception:
        return False


def _compute_phash(img_path: Path):
    try:
        with Image.open(img_path) as im:
            return imagehash.phash(im)
    except Exception:
        return None


def _near_duplicate(h: imagehash.ImageHash, buckets: Dict[str, List[Tuple[imagehash.ImageHash, Path]]], threshold: int) -> bool:
    if h is None:
        return False
    key = str(h)[:12]
    if key not in buckets:
        return False
    for hh, _ in buckets[key]:
        if h - hh <= threshold:
            return True
    return False


def _add_hash(h: imagehash.ImageHash, img_path: Path, buckets: Dict[str, List[Tuple[imagehash.ImageHash, Path]]]):
    if h is None:
        return
    key = str(h)[:12]
    buckets.setdefault(key, []).append((h, img_path))


def _load_model(weights: str | None = None) -> YOLOWorld:
    m = YOLOWorld(weights or "yolov8s-worldv2.pt")
    m.set_classes(["eyeglasses", "glasses", "sunglasses"])
    return m


def _detect_glasses(model: YOLOWorld, img_path: Path):
    res = model(str(img_path))
    if not res:
        return False, []
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return False, []
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    detections = []
    for i in range(len(boxes)):
        detections.append((int(cls[i]), float(conf[i]), boxes[i]))
    has = any(c >= 0 for c in cls)
    return has, detections


def _xyxy_to_yolo(xyxy: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    return xc / w, yc / h, bw / w, bh / h


def filter_dataset(src: Path, out: Path, min_w: int, min_h: int, max_bytes: int, dedupe_threshold: int, sample_limit: int | None, weights: str | None):
    out_images = out / "images"
    out_labels = out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Tuple[imagehash.ImageHash, Path]]] = {}
    model = _load_model(weights)
    imgs = _iter_images(src)
    if sample_limit:
        imgs = imgs[:sample_limit]

    random.shuffle(imgs)
    total_bytes = 0
    class_bytes = {"wear": 0, "no_wear": 0}
    target_per_class = max_bytes // 2
    class_counts = {"wear": 0, "no_wear": 0}
    resolution_skips = 0
    dedupe_skips = 0
    capacity_skips = 0
    w_values = []
    h_values = []

    for img_path in imgs:
        if not _resolution_ok(img_path, min_w, min_h):
            resolution_skips += 1
            continue
        hval = _compute_phash(img_path)
        if _near_duplicate(hval, buckets, dedupe_threshold):
            dedupe_skips += 1
            continue

        has, dets = _detect_glasses(model, img_path)
        cls_name = "wear" if has else "no_wear"
        size = img_path.stat().st_size
        if class_bytes[cls_name] + size > target_per_class:
            capacity_skips += 1
            continue
        if total_bytes + size > max_bytes:
            capacity_skips += 1
            break

        _add_hash(hval, img_path, buckets)
        dst_img = out_images / img_path.name
        shutil.copy2(img_path, dst_img)
        total_bytes += size
        class_bytes[cls_name] += size
        class_counts[cls_name] += 1
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            w_values.append(w)
            h_values.append(h)
        except Exception:
            pass

        if has and dets:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 1, 1
            lines = []
            for cid, conf, xyxy in dets:
                x, y, bw, bh = _xyxy_to_yolo(xyxy, w, h)
                lines.append(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
            (out_labels / (img_path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        else:
            (out_labels / (img_path.stem + ".txt")).write_text("", encoding="utf-8")

    stats = {
        "total_selected_images": class_counts["wear"] + class_counts["no_wear"],
        "total_bytes": total_bytes,
        "class_bytes": class_bytes,
        "class_counts": class_counts,
        "resolution_skips": resolution_skips,
        "dedupe_skips": dedupe_skips,
        "capacity_skips": capacity_skips,
        "width_min": int(min(w_values)) if w_values else 0,
        "width_max": int(max(w_values)) if w_values else 0,
        "width_mean": float(np.mean(w_values)) if w_values else 0.0,
        "height_min": int(min(h_values)) if h_values else 0,
        "height_max": int(max(h_values)) if h_values else 0,
        "height_mean": float(np.mean(h_values)) if h_values else 0.0,
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def split_dataset(filtered_dir: Path, out_dir: Path, train_ratio: float, val_ratio: float):
    images = list((filtered_dir / "images").glob("*"))
    labels_dir = filtered_dir / "labels"
    pairs = []
    for img in images:
        lab = labels_dir / (img.stem + ".txt")
        pairs.append((img, lab))
    def is_wear(lab_path: Path) -> bool:
        try:
            txt = lab_path.read_text(encoding="utf-8")
            return len(txt.strip()) > 0
        except Exception:
            return False
    wear = [p for p in pairs if is_wear(p[1])]
    no_wear = [p for p in pairs if not is_wear(p[1])]
    random.shuffle(wear)
    random.shuffle(no_wear)
    def stratified_split(group: List[Tuple[Path, Path]]):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]
        return train, val, test
    w_tr, w_va, w_te = stratified_split(wear)
    n_tr, n_va, n_te = stratified_split(no_wear)
    splits = {
        "train": w_tr + n_tr,
        "val": w_va + n_va,
        "test": w_te + n_te,
    }
    for split, items in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            shutil.copy2(img, out_dir / split / "images" / img.name)
            shutil.copy2(lab, out_dir / split / "labels" / lab.name)


def evaluate_baseline(split_dir: Path) -> Dict[str, float]:
    images = list((split_dir / "images").glob("*"))
    labels_dir = split_dir / "labels"
    model = _load_model()
    tp = 0
    fp = 0
    fn = 0
    correct_cls = 0
    total = 0
    for img in images:
        lab = labels_dir / (img.stem + ".txt")
        gt_has = False
        try:
            txt = lab.read_text(encoding="utf-8").strip()
            gt_has = len(txt) > 0
        except Exception:
            gt_has = False
        has, dets = _detect_glasses(model, img)
        pred_has = has
        if gt_has and pred_has:
            tp += 1
        elif gt_has and not pred_has:
            fn += 1
        elif not gt_has and pred_has:
            fp += 1
        if gt_has == pred_has:
            correct_cls += 1
        total += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    acc = correct_cls / total if total > 0 else 0.0
    return {"precision": precision, "recall": recall, "accuracy": acc}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    f = sub.add_parser("filter")
    f.add_argument("--src", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--min_w", type=int, default=300)
    f.add_argument("--min_h", type=int, default=300)
    f.add_argument("--max_bytes", type=int, default=500_000_000)
    f.add_argument("--dedupe_threshold", type=int, default=5)
    f.add_argument("--sample_limit", type=int, default=None)
    f.add_argument("--weights", type=str, default=None)

    s = sub.add_parser("split")
    s.add_argument("--filtered", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--train_ratio", type=float, default=0.7)
    s.add_argument("--val_ratio", type=float, default=0.2)

    e = sub.add_parser("evaluate")
    e.add_argument("--split", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "filter":
        filter_dataset(args.src, args.out, args.min_w, args.min_h, args.max_bytes, args.dedupe_threshold, args.sample_limit, args.weights)
    elif args.cmd == "split":
        split_dataset(args.filtered, args.out, args.train_ratio, args.val_ratio)
    elif args.cmd == "evaluate":
        metrics = evaluate_baseline(args.split)
        print(metrics)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

