import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 1.1  Исследование датасета
# ----------------------------------------------------------------------

def inspect_image(path: Path) -> dict:
    """Быстрая проверка размера файла."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Не удалось прочитать {path}")
    h, w = img.shape[:2]
    info = {
        "path": str(path),
        "height": h,
        "width": w,
        "megapixels": round(h * w / 1e6, 2),
    }
    return info


def scan_dataset(input_dir: Path) -> list:
    """Сканирует папку и печатает сводку по размерам изображений."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    report = [inspect_image(p) for p in files]
    for r in report:
        print(f"{r['path']:<40} {r['width']}x{r['height']}  ({r['megapixels']} MP)")
    return report


# ----------------------------------------------------------------------
# Конвертация цветной разметки (линии) в бинарную маску площади
# ----------------------------------------------------------------------

def _perimeter_param(x: int, y: int, w: int, h: int):
    """Unrolls the rectangle border into a 1D coordinate (0..2*(w+h)), clockwise."""
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    
    if y == 0:
        return x
    if x == w - 1:
        return w + y
    if y == h - 1:
        return w + h + (w - 1 - x)
    if x == 0:
        return 2 * w + h + (h - 1 - y)
    return None


def _xy_from_param(t: float, w: int, h: int):
    """Inverse of _perimeter_param."""
    t = t % (2 * (w + h))
    if t < w:
        return int(round(t)), 0
    t -= w
    if t < h:
        return w - 1, int(round(t))
    t -= h
    if t < w:
        return w - 1 - int(round(t)), h - 1
    t -= w
    return 0, h - 1 - int(round(t))


def _rasterize_border_arc(t_start: float, t_end: float, w: int, h: int):
    """Border pixels walking forward (increasing t, mod perimeter) from t_start to t_end."""
    L = 2 * (w + h)
    length = (t_end - t_start) % L
    pts = []
    for i in range(int(length) + 1):
        pts.append(_xy_from_param(t_start + i, w, h))
    return pts


def extract_mask_from_lines(
    image_bgr: np.ndarray,
    lower_hsv = (80, 30, 40),
    upper_hsv = (175, 255, 255),
    close_kernel: int = 5,
    min_area: int = 200,
    cluster_gap: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """Превращает разметку в бинарную маску площади (целиком для изображения)."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    line_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    n_labels, labels = cv2.connectedComponents(line_mask, connectivity=8)
    augmented_wall = line_mask.copy()
    ambiguous_components = []

    for lbl in range(1, n_labels):
        comp = (labels == lbl)
        if comp.sum() < min_area:
            continue

        ys, xs = np.where(comp)
        on_border = (ys == 0) | (ys == h - 1) | (xs == 0) | (xs == w - 1)
        if not on_border.any():
            continue  

        ts = sorted(_perimeter_param(int(x), int(y), w, h) for x, y in zip(xs[on_border], ys[on_border]))

        clusters = [[ts[0]]]
        for t in ts[1:]:
            if t - clusters[-1][-1] <= cluster_gap:
                clusters[-1].append(t)
            else:
                clusters.append([t])
        if len(clusters) > 1 and (2 * (w + h) - clusters[-1][-1] + clusters[0][0]) <= cluster_gap:
            clusters[0] = clusters.pop() + clusters[0]

        touch_ts = sorted(float(np.mean(c)) for c in clusters)

        if len(touch_ts) % 2 != 0:
            ambiguous_components.append({"label": lbl, "n_touches": len(touch_ts), "area": int(comp.sum())})
            if verbose:
                print(f"[ambiguous] component {lbl}: {len(touch_ts)} touch points (odd)")
            #continue

        gaps = []
        for i in range(len(touch_ts)):
            t0 = touch_ts[i]
            t1 = touch_ts[(i + 1) % len(touch_ts)]
            gaps.append((t1 - t0) % (2 * (w + h)))

        background_gap_idx = int(np.argmax(gaps))

        for i in range(len(touch_ts)):
            is_background_gap = (i % 2 == background_gap_idx % 2)
            if is_background_gap:
                continue
            t0, t1 = touch_ts[i], touch_ts[(i + 1) % len(touch_ts)]
            for (px, py) in _rasterize_border_arc(t0, t1, w, h):
                safe_py = max(0, min(py, h - 1))
                safe_px = max(0, min(px, w - 1))
                augmented_wall[safe_py, safe_px] = 255

    n2, labels2 = cv2.connectedComponents(cv2.bitwise_not(augmented_wall), connectivity=4)
    filled = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, n2):
        comp = (labels2 == lbl)
        if comp.sum() < min_area:
            continue
        ys, xs = np.where(comp)
        touches_border = (ys == 0).any() or (ys == h - 1).any() or (xs == 0).any() or (xs == w - 1).any()
        if touches_border:
            continue  
        filled[comp] = 255

    return filled


def find_ambiguous_regions(image_bgr: np.ndarray, lower_hsv=(80, 30, 40), upper_hsv=(175, 255, 255),
                           close_kernel: int = 5, min_area: int = 200, cluster_gap: int = 5):
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    line_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    n_labels, labels = cv2.connectedComponents(line_mask, connectivity=8)

    regions = []
    for lbl in range(1, n_labels):
        comp = (labels == lbl)
        if comp.sum() < min_area:
            continue
        ys, xs = np.where(comp)
        on_border = (ys == 0) | (ys == h - 1) | (xs == 0) | (xs == w - 1)
        if not on_border.any():
            continue
        ts = sorted(_perimeter_param(int(x), int(y), w, h) for x, y in zip(xs[on_border], ys[on_border]))
        clusters = [[ts[0]]]
        for t in ts[1:]:
            if t - clusters[-1][-1] <= cluster_gap:
                clusters[-1].append(t)
            else:
                clusters.append([t])
        if len(clusters) % 2 != 0:
            regions.append({
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "n_touches": len(clusters),
            })
    return regions


# ----------------------------------------------------------------------
# CLAHE нормализация освещения
# ----------------------------------------------------------------------

def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------
# Train/val split по исходным изображениям
# ----------------------------------------------------------------------

def split_by_source(file_ids: list, val_ratio: float = 0.2, seed: int = 42):
    ids = sorted(set(file_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio)) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


# ----------------------------------------------------------------------
# Основной pipeline
# ----------------------------------------------------------------------

def run_pipeline(input_dir: str, output_dir: str, val_ratio: float, seed: int = 42):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    print("=== 1.1 Сканирование датасета ===")
    report = scan_dataset(input_dir)

    file_ids = [Path(r["path"]).stem for r in report]
    train_ids, val_ids = split_by_source(file_ids, val_ratio=val_ratio, seed=seed)
    print(f"\nTrain images: {sorted(train_ids)}")
    print(f"Val images:   {sorted(val_ids)}\n")

    print("=== 1.2 Предобработка и сохранение полных изображений ===")
    manifest = []
    files_needing_manual_fix = []

    for r in report:
        path = Path(r["path"])
        file_id = path.stem
        split = "val" if file_id in val_ids else "train"

        image_raw = cv2.imread(str(path))
        ambiguous = find_ambiguous_regions(image_raw)
        if ambiguous:
            files_needing_manual_fix.append({"file": str(path), "regions": ambiguous})

        # Обработка картинки целиком
        image = apply_clahe(image_raw)
        mask = extract_mask_from_lines(image_raw)

        # Формируем имена файлов (сохраняем оригинальные форматы или принудительно .png)
        img_name = f"{file_id}.png"
        mask_name = img_name

        cv2.imwrite(str(output_dir / split / "images" / img_name), image)
        cv2.imwrite(str(output_dir / split / "masks" / mask_name), mask)

        # Подсчет доли объекта на всем изображении
        fg_ratio = float((mask > 0).mean())

        manifest.append({
            "split": split,
            "source": file_id,
            "image": f"{split}/images/{img_name}",
            "mask": f"{split}/masks/{mask_name}",
            "height": image.shape[0],
            "width": image.shape[1],
            "fg_ratio": round(fg_ratio, 4),
        })

        print(f"{file_id}: Изображение и маска сохранены (split={split})")

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Манифест: {output_dir / 'manifest.json'}")

    if files_needing_manual_fix:
        with open(output_dir / "needs_manual_fix.json", "w", encoding="utf-8") as f:
            json.dump(files_needing_manual_fix, f, ensure_ascii=False, indent=2)
        print(f"\n⚠ {len(files_needing_manual_fix)} файл(ов) содержат линии с нечетным числом касаний рамки.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
