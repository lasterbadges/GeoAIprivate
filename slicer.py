import cv2
import os

# 1. Укажите путь к вашей картинке с синими контурами
image_path = r"Photos/ore.JPG"
image = cv2.imread(image_path)

#Выходная папка
output_dir = "crops"
os.makedirs(output_dir, exist_ok=True)



if image is None:
    print("Ошибка: Картинка не найдена!")
else:
    # Высота и ширина картинки
    height, width, _ = image.shape

    # Размер одного слайса
    tile_size = 512

    # Колво файлов
    count = 0

    # Задаем список кординат по верт заранее чтобы избежать выхода за границу
    y_cords = list(range(0, height, tile_size))
    if y_cords[-1] + tile_size > height:
        y_cords[-1] = height - tile_size

    # Задаем список кординат по гориз заранее чтобы избежать выхода за границу
    x_cords = list(range(0, width, tile_size))
    if x_cords[-1] + tile_size > width:
        x_cords[-1] = width - tile_size

    for y in y_cords:
        for x in x_cords:

            # 2. Вырезаем кусочек: от 0 до 512 пикселей по высоте и по ширине
            crop = image[y:y+tile_size, x:x+tile_size]
            # 3. Сохраняем этот кусочек рядом с кодом
            cv2.imwrite(f"{output_dir}/crop_{count}.jpg", crop)
            count += 1


    print("Супер! Маленький тестовый квадрат сохранен как 'test_crop.jpg'")
