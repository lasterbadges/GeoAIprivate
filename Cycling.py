import cv2
import utils
import os
import glob
import random
import numpy

INPUT_DIR = "Master"
OUTPUT_DIR = "OutputDir"
extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.JPG"]
TRAIN_RATIO = 0.8
tile_size = 512
os.makedirs("OutputDir", exist_ok=True)
PrePr = utils.PreProcessingMethods()

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

os.makedirs(os.path.join(OUTPUT_DIR,  "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,  "masks"), exist_ok=True)


image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

# Оставляем только уникальные пути и нормализуем их
image_paths = list(set(os.path.normpath(p) for p in image_paths))

random.seed(42)
random.shuffle(image_paths)
print(image_paths)
split_idx = int(len(image_paths) * TRAIN_RATIO)
datasets = {
    "train": image_paths[:split_idx],
    "val": image_paths[split_idx:]
}

count = 0

for img_path in image_paths:

    image_bgr, image_enhanced_gray, image_hsv = PrePr.preprocess_image(img_path)
    filled_mask = PrePr.create_binary_mask(image_hsv)

    height, width, _ = image_bgr.shape
    # Картинку сохраняем как обычно
    img_save_path = os.path.join(OUTPUT_DIR, "images", f"crop_{count}.png")
    cv2.imwrite(img_save_path, image_bgr)

    # Сохраняем маску как нормальный видимый PNG
    mask_save_path = os.path.join(OUTPUT_DIR, "masks", f"crop_{count}.png")
    cv2.imwrite(mask_save_path, filled_mask)
    count += 1
"""
    # Твой слайсер
    y_cords = list(range(0, height, tile_size))
    if y_cords[-1] + tile_size > height:
        y_cords[-1] = height - tile_size

    x_cords = list(range(0, width, tile_size))
    if x_cords[-1] + tile_size > width:
        x_cords[-1] = width - tile_size

    for y in y_cords:
        for x in x_cords:
            crop_img = image_bgr[y: y + tile_size, x: x + tile_size]
            crop_mask_layers = filled_mask[y: y + tile_size, x: x + tile_size]

            # Картинку сохраняем как обычно
            img_save_path = os.path.join(OUTPUT_DIR, "images", f"crop_{count}.png")
            cv2.imwrite(img_save_path, crop_img)



            # Сохраняем маску как нормальный видимый PNG
            mask_save_path = os.path.join(OUTPUT_DIR, "masks", f"crop_{count}.png")
            cv2.imwrite(mask_save_path, crop_mask_layers)

print(f"Датасет успешно собран модульным пайплайном! Патчей: {count}")"""