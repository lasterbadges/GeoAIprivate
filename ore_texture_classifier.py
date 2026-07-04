from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


TEXTURE_CLASSIFIER_PATH = Path("models/ore_texture_classifier.pth")
CLASS_NAMES = {
    0: "Труднообогатимая",
    1: "Рядовая",
}


def read_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def normalize_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim != 3:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    image_bgr = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    means = np.mean(image_bgr.reshape(-1, 3).astype(np.float32), axis=0) + 1e-6
    target = float(np.mean(means))
    image_bgr = np.clip(image_bgr.astype(np.float32) * (target / means), 0, 255)
    return image_bgr.astype(np.uint8)


def build_texture_model(arch: str = "efficientnet_b0", weights=None):
    import torch
    from torchvision import models

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)
        return model

    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model

    raise ValueError(f"Неизвестная архитектура: {arch}")


def _tile_score(tile_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    return float(gray.std() + 0.35 * saturation.std())


def make_inference_tiles(
    image_bgr: np.ndarray,
    crop_size: int = 448,
    max_tiles: int = 36,
) -> list[np.ndarray]:
    image_bgr = normalize_bgr(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    if min(h, w) < crop_size:
        side = max(h, w, crop_size)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = image_rgb
        image_rgb = canvas
        h, w = image_rgb.shape[:2]

    rows = max(2, min(6, math.ceil(h / crop_size)))
    cols = max(2, min(6, math.ceil(w / crop_size)))
    ys = np.linspace(0, max(0, h - crop_size), rows).astype(int)
    xs = np.linspace(0, max(0, w - crop_size), cols).astype(int)

    candidates: list[tuple[float, np.ndarray]] = []
    for y in ys:
        for x in xs:
            tile = image_rgb[y:y + crop_size, x:x + crop_size]
            if tile.shape[:2] != (crop_size, crop_size):
                tile = cv2.resize(tile, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            candidates.append((_tile_score(tile), tile))

    center = image_rgb[
        max(0, (h - crop_size) // 2):max(0, (h - crop_size) // 2) + crop_size,
        max(0, (w - crop_size) // 2):max(0, (w - crop_size) // 2) + crop_size,
    ]
    if center.shape[:2] == (crop_size, crop_size):
        candidates.append((_tile_score(center) + 1.0, center))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [tile for _, tile in candidates[:max_tiles]]


def _load_checkpoint(model_path: str | Path, device):
    import torch

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        arch = checkpoint.get("arch", "efficientnet_b0")
        model = build_texture_model(arch=arch, weights=None)
        model.load_state_dict(checkpoint["state_dict"])
        threshold = float(checkpoint.get("regular_threshold", 0.5))
        image_size = int(checkpoint.get("image_size", 224))
        return model, threshold, image_size

    model = build_texture_model(arch="resnet18", weights=None)
    model.load_state_dict(checkpoint)
    return model, 0.5, 224


def predict_ore_texture(
    image_bgr_or_path: np.ndarray | str | Path,
    model_path: str | Path = TEXTURE_CLASSIFIER_PATH,
    device=None,
    max_tiles: int = 36,
) -> dict:
    import torch
    from torchvision import transforms

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Модель классификации текстуры не найдена: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_bgr = read_bgr(image_bgr_or_path) if isinstance(image_bgr_or_path, (str, Path)) else image_bgr_or_path
    model, threshold, image_size = _load_checkpoint(model_path, device)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tiles = make_inference_tiles(image_bgr, max_tiles=max_tiles)
    if not tiles:
        raise ValueError("Не удалось получить тайлы для классификации")

    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(tiles), 32):
            batch = torch.stack([transform(tile) for tile in tiles[start:start + 32]]).to(device)
            prob_regular = torch.softmax(model(batch), dim=1)[:, 1].detach().cpu().numpy()
            probs.extend(float(p) for p in prob_regular)

    regular_probability = float(np.median(probs))
    predicted_label = 1 if regular_probability >= 0.5 else 0
    confidence = regular_probability if predicted_label == 1 else 1.0 - regular_probability

    return {
        "class": CLASS_NAMES[predicted_label],
        "label": predicted_label,
        "confidence": round(confidence * 100.0, 1),
        "regular_probability": round(regular_probability * 100.0, 1),
        "complex_probability": round((1.0 - regular_probability) * 100.0, 1),
        "decision_threshold": 0.5,
        "tile_count": len(tiles),
        "tile_probabilities": probs,
    }


def iter_image_files(paths: Iterable[Path]) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".tga"}
    return [path for path in paths if path.is_file() and path.suffix.lower() in suffixes]
