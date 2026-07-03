import cv2
import numpy
# Подготовка изображеняи цветокорекция clahe возваращает три копии в разных цвет пространствах
def preprocess_image(image_path:str):

    # 1. Загрузка изображения в формате BGR
    image_array = numpy.fromfile(image_path, dtype=numpy.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")

    # 2. Перевод в оттенки серого
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Настройка и применение адаптивного выравнивания контраста (CLAHE)
    # clipLimit=3.0 — порог ограничения контраста. Чем выше, тем контрастнее (и больше шума).
    # tileGridSize=(8, 8) — размер сетки, на которую бьется картинка для локального анализа.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image_enhanced_gray = clahe.apply(image_gray)

    # 4. Перевод оригинального BGR в HSV для последующего выделения синих контуров
    # cv2.COLOR_BGR2HSV меняет представление с каналов B-G-R на Тон-Насыщенность-Яркость.
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Возвращаем кортеж из трех обработанных матриц NumPy
    return image_bgr, image_enhanced_gray, image_hsv


def create_binary_mask(image_hsv):
    # Выделяет синие контуры разметки из HSV-изображения, создавая маску зерен.


    # 1. Задаем границы синего цвета в пространстве OpenCV HSV
    # H (Тон) для синего: ~100-140 (из 180)
    # S (Насыщенность) и V (Яркость): отсекаем слишком блеклые или темные тона
    lower_blue = numpy.array([100, 50, 50])
    upper_blue = numpy.array([140, 255, 255])

    # 2. Создаем первичную маску синих линий
    mask_blue_lines = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # 3. Заполнение внутри синих линий
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_closed = cv2.morphologyEx(mask_blue_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    # MORPH_CLOSE сначала расширяет белые зоны, соединяя разрывы, а потом сужает обратно
    mask_closed = cv2.morphologyEx(mask_blue_lines, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Поиск контуров
    # cv2.RETR_EXTERNAL — ищем только самые внешние контуры (игнорируем вложенные дырки)
    # cv2.CHAIN_APPROX_SIMPLE — сжимает горизонтальные, вертикальные и диагональные сегменты
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5.Создание черной маски
    filled_mask = numpy.zeros_like(mask_closed)

    # 6. Заливаем найденные контуры изнутри
    # -1 — означает «отрисовать ВСЕ найденные контуры»
    # 255 — цвет заливки (белый)
    # thickness=cv2.FILLED — ключевой флаг, который приказывает залить контур полностью, а не просто обвести
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    return filled_mask

def classify_ore_segments(filled_mask, image_enhanced_gray, brightness_threshold=120, talc_threshold=30):
    """
        Разделяет бинарную маску на 4 класса на основе яркости и текстуры в Grayscale.

        Параметры:
        - filled_mask: бинарная маска зерен из Шага 2 (0 или 255)
        - image_enhanced_gray: ч/б картинка после CLAHE из Шага 1
        - brightness_threshold: порог средней яркости, разделяющий Рядовую (1) и Труднообогатимую (2) руду
        - talc_threshold: порог яркости для поиска талька (всё, что темнее этого значения в зоне породы)
    """

    # 1. Инициализируем итоговую многоклассовую маску
    # Заполняем её нулями, так как Класс 0 (Порода) — это наше состояние по умолчанию.
    # Тип данных обязательно np.uint8, чтобы маску можно было сохранить как обычное изображение.
    final_mask = numpy.zeros_like(filled_mask, dtype=numpy.uint8)

    # 2. Изолируем зерна для поштучного анализа
    # Снова находим контуры, чтобы обрабатывать каждое зерно (камень) независимо от других.
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # Создаем временную маску ТОЛЬКО для одного текущего зерна
        single_grain_mask = numpy.zeros_like(filled_mask)
        cv2.drawContours(single_grain_mask, contours, i, 255, thickness=cv2.FILLED)

        # Магия NumPy: извлекаем яркость пикселей оригинального ч/б кадра,
        # но только в тех координатах, где на single_grain_mask стоит 255 (наше зерно).
        grain_pixels = image_enhanced_gray[single_grain_mask == 255]

        # Если контур микроскопический (шум), пропускаем его, чтобы избежать ошибок деления на ноль
        if len(grain_pixels) == 0:
            continue

        # Считаем среднюю арифметическую яркость этого конкретного куска руды
        mean_brightness = numpy.mean(grain_pixels)

        # Классифицируем сорт руды по средней яркости:
        if mean_brightness >= brightness_threshold:
            # Если камень светлый — это Класс 1 (Рядовая руда)
            final_mask[single_grain_mask == 255] = 1
        else:
            # Если камень темный — это Класс 2 (Труднообогатимая руда)
            final_mask[single_grain_mask == 255] = 2

    # 3. Поиск Класса 3 (Тальк) в зоне породы
    talc_condition = (filled_mask == 0) & (image_enhanced_gray < talc_threshold)
    final_mask[talc_condition] = 3

    return final_mask
