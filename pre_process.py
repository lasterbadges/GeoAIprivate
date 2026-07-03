import os
import glob
import random
import cv2
import numpy

from utils import preprocess_image, create_binary_mask, classify_ore_segments

INPUT_DIR = "InputDir"
OUTPUT_DIR = "OutputDir"
extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.JPG"]
#TRAIN_RATIO = 0.8
tile_size = 512
os.makedirs("OutputDir", exist_ok=True)


os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

os.makedirs(os.path.join(OUTPUT_DIR,  "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,  "masks"), exist_ok=True)


image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

# Оставляем только уникальные пути и нормализуем их
image_paths = list(set(os.path.normpath(p) for p in image_paths))
print(image_paths)
random.seed(42)
random.shuffle(image_paths)

#split_idx = int(len(image_paths) * TRAIN_RATIO)
#datasets = {
    #"train": image_paths[:split_idx],
    #"val": image_paths[split_idx:]
#}

count = 0

for img_path in image_paths:

    image_bgr, image_enhanced_gray, image_hsv = preprocess_image(img_path)
    filled_mask = create_binary_mask(image_hsv)
    final_mask = classify_ore_segments(filled_mask, image_enhanced_gray)

    height, width, _ = image_bgr.shape

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
            crop_mask_layers = final_mask[y: y + tile_size, x: x + tile_size]

            # Картинку сохраняем как обычно
            img_save_path = os.path.join(OUTPUT_DIR, "images", f"crop_{count}.png")
            cv2.imwrite(img_save_path, crop_img)

            # --- РАСКРАШИВАЕМ 2D МАСКУ В ЦВЕТНУЮ КАРТИНКУ ---
            h_crop, w_crop = crop_mask_layers.shape[:2]
            color_mask = numpy.zeros((h_crop, w_crop, 3), dtype=numpy.uint8)

            # Проверяем значения пикселей в 2D массиве
            # Значение == 1: Руда -> серый (128, 128, 128)
            color_mask[crop_mask_layers == 1] = [128, 128, 128]

            # Значение == 2: Тальк -> белый (255, 255, 255)
            color_mask[crop_mask_layers == 2] = [255, 255, 255]

            # Значение == 3: Границы/срастания -> синий BGR (255, 0, 0)
            color_mask[crop_mask_layers == 3] = [255, 0, 0]

            # Сохраняем маску как нормальный видимый PNG
            mask_save_path = os.path.join(OUTPUT_DIR, "masks", f"crop_{count}.png")
            cv2.imwrite(mask_save_path, color_mask)

print(f"Датасет успешно собран модульным пайплайном! Патчей: {count}")