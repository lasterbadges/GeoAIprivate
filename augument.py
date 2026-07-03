import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

IMAGE_DIR = "dataset/train/images"
MASK_DIR = "dataset/train/masks"

# Сколько комбинированных вариантов сделать из КАЖДОЙ картинки ( 5 - это примерно 200 разных шлифов)
NUM_VARIANTS = 100

def safe_imread(file_path, flags=cv2.IMREAD_UNCHANGED):
    try:
        nparr = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(nparr, flags)
    except Exception as e:
        print(f"\nОшибка чтения: {file_path} -> {e}")
        return None

def safe_imwrite(file_path, img):
    try:
        ext = os.path.splitext(file_path)[1]
        result, nparr = cv2.imencode(ext, img)
        if result:
            nparr.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"\nОшибка записи: {file_path} -> {e}")
        return False

def get_mega_pipeline():
    """
    Создает один мощный конвейер, где ВСЕ эффекты 
    применяются ОДНОВРЕМЕННО и перемешиваются
    """
    return A.Compose([
        # 1. ГЕОМЕТРИЯ (Применится всегда: крутим и зеркалим)
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 2. СВЕТ И КОНТРАСТ (С вероятностью 70% изменит освещение)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.CLAHE(clip_limit=3.0, p=0.5),
        
        # 3. ОПТИКА И ШУМ (С вероятностью 60% добавит размытие ИЛИ шум камеры)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(15.0, 50.0), p=1.0),
        ], p=0.6),
        
        # 4. ДЕФЕКТЫ ПОЛИРОВКИ (С вероятностью 50% бахнет мелкие царапины/грязь)
        A.CoarseDropout(max_holes=5, max_height=12, max_width=12, fill_value=0, p=0.5)
    ])

def run_mega_augmentation():
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
        print("Проверь пути к папкам!")
        return

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    # Игнорируем уже созданные ранее копии, работаем только с оригиналами
    image_files = [f for f in image_files if "_variant" not in f]
    
    print(f"Исходных шлифов найдено: {len(image_files)}")
    print(f"Каждый будет размножен на {NUM_VARIANTS} уникальных комбинированных копий.")
    
    mega_pipeline = get_mega_pipeline()

    for file_name in tqdm(image_files):
        img_path = os.path.normpath(os.path.join(IMAGE_DIR, file_name))
        mask_path = os.path.normpath(os.path.join(MASK_DIR, file_name))

        if not os.path.exists(mask_path):
            name_without_ext = os.path.splitext(file_name)[0]
            mask_path = os.path.normpath(os.path.join(MASK_DIR, name_without_ext + ".png"))
            if not os.path.exists(mask_path):
                continue

        img = safe_imread(img_path, cv2.IMREAD_COLOR)
        mask = safe_imread(mask_path, cv2.IMREAD_UNCHANGED)

        if img is None or mask is None:
            continue

        name, ext = os.path.splitext(file_name)
        mask_ext = os.path.splitext(os.path.basename(mask_path))[1]

        # Генерируем N тяжелых комбинированных вариантов
        for i in range(1, NUM_VARIANTS + 1):
            # Тут магия: запускаем пайплайн, он рандомно смешивает геометрию и физику
            augmented = mega_pipeline(image=img, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            new_img_name = f"{name}_variant{i}{ext}"
            new_mask_name = f"{name}_variant{i}{mask_ext}"

            safe_imwrite(os.path.normpath(os.path.join(IMAGE_DIR, new_img_name)), aug_img)
            safe_imwrite(os.path.normpath(os.path.join(MASK_DIR, new_mask_name)), aug_mask)

    print("\n[КОМБО-УСПЕХ] Все изображения размножены и упакованы эффектами!")

if __name__ == "__main__":
    run_mega_augmentation()