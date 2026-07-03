"""
🔬 Мастер-скрипт подготовки данных «Все в одном».
Запускает полный цикл препроцессинга за один шаг:
1. Организация исходного датасета (перенос и сортировка).
2. Автоматическое извлечение масок талька из синей разметки.
3. Локальная нормализация яркости (CLAHE + виньетирование).
4. Разделение данных по задачам обучения (классификация / сегментация).

Запуск:
    python preprocessing/prepare_all.py
"""

import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from loguru import logger

# Вычисляем корень проекта динамически
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Задача 3. Скажи мне, кто твой шлиф"

# Пути для промежуточных данных
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEST_TALC = PROCESSED_DIR / "talc"
DEST_REGULAR = PROCESSED_DIR / "regular"
DEST_COMPLEX = PROCESSED_DIR / "complex"
DEST_ANNOTATIONS = PROCESSED_DIR / "talc_annotations"
MASKS_DIR = PROCESSED_DIR / "talc_masks"

# Пути для нормализации и финального деления
NORM_DIR = BASE_DIR / "data" / "normalized"
CLASS_DIR = BASE_DIR / "data" / "train_classification"
SEG_DIR = BASE_DIR / "data" / "train_segmentation"


# ============================================================
# Вспомогательные функции чтения/записи
# ============================================================
def imread_unicode(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    try:
        nparr = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(nparr, flags)
    except Exception as e:
        logger.error(f"Ошибка чтения {path}: {e}")
        return None


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    try:
        ext = Path(path).suffix
        is_success, im_buf_arr = cv2.imencode(ext, img, params)
        if is_success:
            im_buf_arr.tofile(str(path))
            return True
        return False
    except Exception as e:
        logger.error(f"Ошибка записи {path}: {e}")
        return False


# ============================================================
# ШАГ 1: Организация файлов
# ============================================================
def copy_files(src_dir: Path, dest_dir: Path, suffix: str):
    if not src_dir.exists():
        logger.warning(f"Исходная папка не найдена: {src_dir}")
        return 0
    copied = 0
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            dest_file = dest_dir / f"{f.stem}{suffix}{f.suffix}"
            try:
                shutil.copy2(f, dest_file)
                copied += 1
            except Exception as e:
                logger.error(f"Ошибка копирования {f.name}: {e}")
    return copied


def organize_dataset():
    logger.info("--- Шаг 1: Организация структуры папок ---")
    for folder in [DEST_TALC, DEST_REGULAR, DEST_COMPLEX, DEST_ANNOTATIONS]:
        folder.mkdir(parents=True, exist_ok=True)
        
    c1_dir = DATA_DIR / "Фото руд по сортам. ч1"
    c2_dir = DATA_DIR / "Фото руд по сортам. ч2"
    
    total = 0
    total += copy_files(c1_dir / "Рядовые руды", DEST_REGULAR, "_p1")
    total += copy_files(c2_dir / "рядовые", DEST_REGULAR, "_p2")
    
    total += copy_files(c1_dir / "Труднообогатимые руды", DEST_COMPLEX, "_p1")
    total += copy_files(c2_dir / "тонкие", DEST_COMPLEX, "_p2")
    
    total += copy_files(c1_dir / "Оталькованные руды", DEST_TALC, "_p1")
    total += copy_files(c2_dir / "оталькованные", DEST_TALC, "_p2")
    
    total += copy_files(c1_dir / "Оталькованные руды" / "Области оталькования", DEST_ANNOTATIONS, "_p1")
    
    logger.info(f"Структурировано файлов: {total}")


# ============================================================
# ШАГ 2: Извлечение масок
# ============================================================
def extract_blue_contour(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([100, 150, 100]), np.array([135, 255, 255]))
    b, g, r = cv2.split(image_bgr)
    rgb_blue = ((b.astype(int) - r.astype(int) > 80) &
                (b.astype(int) - g.astype(int) > 80) &
                (b > 150)).astype(np.uint8) * 255
    return cv2.bitwise_or(blue_mask, rgb_blue)


def contour_to_filled_mask(contour_mask: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(contour_mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def extract_masks():
    logger.info("--- Шаг 2: Извлечение масок талька ---")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    annot_files = sorted(DEST_ANNOTATIONS.glob("*.JPG")) + sorted(DEST_ANNOTATIONS.glob("*.jpg"))
    
    if not annot_files:
        logger.error(f"Аннотации не найдены в {DEST_ANNOTATIONS}")
        return
        
    stats = 0
    for annot_file in annot_files:
        orig_file = DEST_TALC / annot_file.name
        mask_file = MASKS_DIR / f"{annot_file.stem}_mask.png"
        try:
            annot = imread_unicode(str(annot_file))
            if orig_file.exists():
                orig = imread_unicode(str(orig_file))
                if orig is not None and orig.shape == annot.shape:
                    diff = cv2.absdiff(annot, orig)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, diff_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
                    combined = cv2.bitwise_or(diff_mask, extract_blue_contour(annot))
                else:
                    combined = extract_blue_contour(annot)
            else:
                combined = extract_blue_contour(annot)
                
            mask = contour_to_filled_mask(combined)
            imwrite_unicode(str(mask_file), mask)
            stats += 1
        except Exception as e:
            logger.error(f"Ошибка извлечения для {annot_file.name}: {e}")
            
    logger.info(f"Успешно сгенерировано масок: {stats}")


# ============================================================
# ШАГ 3: Нормализация изображений (CLAHE + Виньетки)
# ============================================================
def normalize_img(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 0.1
    small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    small_sigma = max(small.shape[:2]) // 4
    small_sigma = small_sigma if small_sigma % 2 == 1 else small_sigma + 1
    small_blur = cv2.GaussianBlur(small.astype(np.float32), (small_sigma, small_sigma), 0)
    illumination = cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)
    corrected = img_bgr.astype(np.float32) * illumination.mean() / (illumination + 1e-6)
    img_vignette = np.clip(corrected, 0, 255).astype(np.uint8)
    
    lab = cv2.cvtColor(img_vignette, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_norm, a, b]), cv2.COLOR_LAB2BGR)


def run_normalization():
    logger.info("--- Шаг 3: Нормализация яркости и контраста ---")
    classes = ["talc", "regular", "complex"]
    for cls in classes:
        (NORM_DIR / cls).mkdir(parents=True, exist_ok=True)
        files = list((PROCESSED_DIR / cls).glob("*"))
        logger.info(f"Нормализация класса {cls}: {len(files)} файлов...")
        for f in files:
            img = imread_unicode(str(f))
            if img is not None:
                norm = normalize_img(img)
                imwrite_unicode(str(NORM_DIR / cls / f.name), norm)


# ============================================================
# ШАГ 4: Разделение данных по задачам
# ============================================================
def split_tasks():
    logger.info("--- Шаг 4: Разделение на задачи классификации и сегментации ---")
    for sub in ["regular", "complex", "talc"]:
        (CLASS_DIR / sub).mkdir(parents=True, exist_ok=True)
    (SEG_DIR / "images").mkdir(parents=True, exist_ok=True)
    (SEG_DIR / "masks").mkdir(parents=True, exist_ok=True)
    
    for category in ["regular", "complex", "talc"]:
        src_cat = NORM_DIR / category
        dest_cat = CLASS_DIR / category
        if src_cat.exists():
            for f in src_cat.glob("*"):
                shutil.copy2(f, dest_cat / f.name)
                
    seg_count = 0
    for mask_path in MASKS_DIR.glob("*_mask.png"):
        orig_name = mask_path.name.replace("_mask.png", ".JPG")
        orig_path = NORM_DIR / "talc" / orig_name
        if not orig_path.exists():
            orig_name = mask_path.name.replace("_mask.png", ".jpg")
            orig_path = NORM_DIR / "talc" / orig_name
            
        if orig_path.exists():
            shutil.copy2(orig_path, SEG_DIR / "images" / orig_path.name)
            shutil.copy2(mask_path, SEG_DIR / "masks" / mask_path.name)
            seg_count += 1
            
    logger.info(f"Классификация подготовлена в {CLASS_DIR}")
    logger.info(f"Сегментация подготовлена в {SEG_DIR} ({seg_count} пар)")


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
def main():
    log_file = BASE_DIR / "data" / "logs" / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="5 MB")
    
    logger.info("============================================================")
    logger.info("🚀 Запуск объединенного препроцессинга данных руд")
    logger.info("============================================================")
    
    if not DATA_DIR.exists():
        logger.error(f"Папка с исходными данными не найдена по пути: {DATA_DIR}")
        logger.info("Пожалуйста, убедитесь, что вы скачали и распаковали архив с Яндекс.Диска!")
        sys.exit(1)
        
    try:
        organize_dataset()
        extract_masks()
        run_normalization()
        split_tasks()
        logger.info("🎉 Подготовка данных завершена на 100%! Всё готово к обучению!")
    except Exception as e:
        logger.exception(f"Критическая ошибка в пайплайне: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
