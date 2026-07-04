import cv2
import numpy
import os

class PreProcessingMethods:
    def __init__(self):
        self.unet_model = None
        self.resnet_model = None

    # Подготовка изображеняи цветокорекция clahe возваращает три копии в разных цвет пространствах
    def preprocess_image(self, image_path):
        # 1. Загрузка изображения в формате BGR
        image_array = numpy.fromfile(str(image_path), dtype=numpy.uint8)
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

    def create_binary_mask(self, image_hsv):
        lower_blue = numpy.array([100, 50, 50])
        upper_blue = numpy.array([140, 255, 255])
        mask_blue_lines = cv2.inRange(image_hsv, lower_blue, upper_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_closed = cv2.morphologyEx(mask_blue_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = numpy.zeros_like(mask_closed)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        return filled_mask

    def segment_sulfides(self, image_enhanced_gray):
        """
        Сегментирует сульфиды на основе Otsu-порога и фильтрации темных пикселей.
        Используется для необработанных (тестовых) изображений без синих линий разметки.
        """
        # Применяем порог Otsu
        _, thresh = cv2.threshold(image_enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Руда/сульфиды должны быть яркими. Отсекаем темную нерудную матрицу.
        # Обычно сульфиды после CLAHE ярче 100-110.
        thresh[image_enhanced_gray < 110] = 0
        
        # Убираем мелкие шумы
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask_opened

    def segment_talc_unet(self, image_bgr, model_path="models/test_unet.pth", device=None):
        """
        Сегментирует тальк с использованием глубокой сети U-Net (ResNet18).
        Применяет тайлинг (нарезку) для работы с большими разрешениями и сшивает результат.
        """
        import torch
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.unet_model is None:
            try:
                import segmentation_models_pytorch as smp
            except ImportError:
                raise ImportError("segmentation_models_pytorch не установлена. Запустите pip install segmentation-models-pytorch")

            # 1. Загрузка U-Net
            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            self.unet_model = model
        else:
            model = self.unet_model

        from preprocessing.tiling import slice_into_tiles, stitch_tiles

        # 2. Нарезка на тайлы
        # Нарезаем исходное изображение на тайлы 512x512 с перекрытием 64
        tiles, coords = slice_into_tiles(image_bgr, tile_size=512, overlap=64)

        pred_tiles = []
        with torch.no_grad():
            for tile in tiles:
                # Переводим в RGB и ресайзим к 256x256 (на которых обучалась сеть)
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_resized = cv2.resize(tile_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
                
                # Подготавливаем тензор [1, 3, 256, 256]
                tensor = torch.from_numpy(tile_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                tensor = tensor.to(device)
                
                # Запуск инференса
                logits = model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0] # [256, 256]
                
                # Порог бинаризации
                pred_tile_resized = (probs > 0.5).astype(numpy.uint8) * 255
                
                # Ресайзим обратно к 512x512
                pred_tile = cv2.resize(pred_tile_resized, (512, 512), interpolation=cv2.INTER_NEAREST)
                pred_tiles.append(pred_tile)

        # 3. Сшиваем тайлы обратно
        stitched_mask = stitch_tiles(pred_tiles, coords, image_bgr.shape[:2], tile_size=512)
        return stitched_mask

    def classify_sulfides_resnet(self, image_bgr, sulfide_mask, device=None, model_path="models/ore_resnet18.pth"):
        """
        Классифицирует сульфиды на обычные (класс 1) и тонкие (класс 2) с помощью ResNet-18 скользящим окном.
        """
        import torch
        from torchvision import models, transforms
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.resnet_model is None:
            # Инициализация ResNet-18
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            self.resnet_model = model
        else:
            model = self.resnet_model

        # Предобработка патчей
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        h, w, _ = image_bgr.shape
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        patch_size = 256
        stride = 128

        y_coords = list(range(0, h - patch_size + 1, stride))
        if y_coords[-1] + patch_size < h:
            y_coords.append(h - patch_size)

        x_coords = list(range(0, w - patch_size + 1, stride))
        if x_coords[-1] + patch_size < w:
            x_coords.append(w - patch_size)

        patches = []
        coords = []

        for y in y_coords:
            for x in x_coords:
                patch = img_rgb[y:y+patch_size, x:x+patch_size]
                patches.append(transform(patch))
                coords.append((y, x))

        # Запуск инференса батчами
        batch_size = 64
        probs = []

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i:i+batch_size]).to(device)
                outputs = model(batch)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs.extend(batch_probs)

        # Карта вероятностей
        prob_map = numpy.zeros((h, w), dtype=numpy.float32)
        count_map = numpy.zeros((h, w), dtype=numpy.float32)

        for (y, x), prob in zip(coords, probs):
            prob_map[y:y+patch_size, x:x+patch_size] += prob[1]
            count_map[y:y+patch_size, x:x+patch_size] += 1.0

        count_map[count_map == 0] = 1.0
        prob_map /= count_map

        final_mask = numpy.zeros_like(sulfide_mask, dtype=numpy.uint8)
        contours, _ = cv2.findContours(sulfide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Классификация зерен
        for i, contour in enumerate(contours):
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            if w_c == 0 or h_c == 0:
                continue

            local_mask = numpy.zeros((h_c, w_c), dtype=numpy.uint8)
            shifted_contour = contour - [x_c, y_c]
            cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

            crop_prob = prob_map[y_c:y_c+h_c, x_c:x_c+w_c]
            grain_probs = crop_prob[local_mask == 255]

            if len(grain_probs) == 0:
                continue

            mean_grain_prob = numpy.mean(grain_probs)
            # Класс 1 (Обычные) при prob >= 0.5, иначе Класс 2 (Тонкие)
            val = 1 if mean_grain_prob >= 0.5 else 2

            final_mask_crop = final_mask[y_c:y_c+h_c, x_c:x_c+w_c]
            final_mask_crop[local_mask == 255] = val

        return final_mask

    def classify_ore_segments(self, filled_mask, image_enhanced_gray, brightness_threshold=120, talc_threshold=30, image_bgr=None, use_unet=True, model_path="models/test_unet.pth", device=None, use_resnet=True, resnet_model_path="models/ore_resnet18.pth"):
        """
        Разделяет изображение на 4 класса:
        - Класс 0 (Порода/Матрица)
        - Класс 1 (Обычные сульфиды / крупная яркая фаза)
        - Класс 2 (Тонкие сульфиды / мелкая/замещенная темная фаза)
        - Класс 3 (Тальк)
        """
        # 1. Проверяем, является ли filled_mask маской разметки (синие линии)
        # Если filled_mask пустая (или почти пустая, менее 1000 пикселей), мы автоматически сегментируем сульфиды
        if numpy.sum(filled_mask > 0) < 1000:
            sulfide_mask = self.segment_sulfides(image_enhanced_gray)
        else:
            sulfide_mask = filled_mask

        # 2. Инициализируем итоговую многоклассовую маску
        final_mask = numpy.zeros_like(sulfide_mask, dtype=numpy.uint8)

        # 3. Сегментируем Тальк (Класс 3)
        talc_mask = None
        # Попробуем запустить U-Net
        if use_unet and image_bgr is not None and os.path.exists(model_path):
            try:
                talc_mask = self.segment_talc_unet(image_bgr, model_path, device)
            except Exception as e:
                # Если упало (например, GPU OOM или проблемы с торчем), используем фолбек
                pass

        if talc_mask is None:
            # Классический фолбек по яркости: темные зоны в нерудной матрице
            talc_mask = (sulfide_mask == 0) & (image_enhanced_gray < talc_threshold)
        else:
            # Приводим к булевой маске
            talc_mask = talc_mask > 127

        # Записываем тальк (Класс 3)
        final_mask[talc_mask] = 3

        # 4. Классифицируем сульфиды на обычные (Класс 1) и тонкие (Класс 2)
        resnet_success = False
        if use_resnet and image_bgr is not None and os.path.exists(resnet_model_path):
            try:
                resnet_mask = self.classify_sulfides_resnet(image_bgr, sulfide_mask, device, resnet_model_path)
                # Переносим классификацию сульфидов, исключая зоны талька
                final_mask[(resnet_mask == 1) & (final_mask != 3)] = 1
                final_mask[(resnet_mask == 2) & (final_mask != 3)] = 2
                resnet_success = True
            except Exception as e:
                # В случае ошибки падаем на классический пороговый метод
                pass

        if not resnet_success:
            # Классический оптимизированный фолбек по яркости с bounding box (ускорение в ~100 раз)
            contours, _ = cv2.findContours(sulfide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                if w_c == 0 or h_c == 0:
                    continue

                local_mask = numpy.zeros((h_c, w_c), dtype=numpy.uint8)
                shifted_contour = contour - [x_c, y_c]
                cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

                crop_gray = image_enhanced_gray[y_c:y_c+h_c, x_c:x_c+w_c]
                grain_pixels = crop_gray[local_mask == 255]

                if len(grain_pixels) == 0:
                    continue

                mean_brightness = numpy.mean(grain_pixels)
                val = 1 if mean_brightness >= brightness_threshold else 2

                final_mask_crop = final_mask[y_c:y_c+h_c, x_c:x_c+w_c]
                final_mask_crop[(local_mask == 255) & (final_mask_crop != 3)] = val

        return final_mask

    def slicer(self, image_path, output_dir):
        image = cv2.imread(image_path)
        output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if image is None:
            print("Ошибка: Картинка не найдена!")
            return

        height, width, _ = image.shape
        tile_size = 512
        count = 0

        y_cords = list(range(0, height, tile_size))
        if y_cords[-1] + tile_size > height:
            y_cords[-1] = height - tile_size

        x_cords = list(range(0, width, tile_size))
        if x_cords[-1] + tile_size > width:
            x_cords[-1] = width - tile_size

        for y in y_cords:
            for x in x_cords:
                crop = image[y:y + tile_size, x:x + tile_size]
                cv2.imwrite(f"{output_dir}/crop_{count}.jpg", crop)
                count += 1

    def mask_and_image_slicer(self, dirmask, dirimage):
        tile_size = 512
        image_bgr, image_enhanced_gray, image_hsv = self.preprocess_image(dirimage)
        filled_mask = self.create_binary_mask(image_hsv)
        final_mask = self.classify_ore_segments(filled_mask, image_enhanced_gray)
        height, width, _ = image_bgr.shape
        
        y_cords = list(range(0, height, tile_size))
        if y_cords[-1] + tile_size > height:
            y_cords[-1] = height - tile_size

        x_cords = list(range(0, width, tile_size))
        if x_cords[-1] + tile_size > width:
            x_cords[-1] = width - tile_size

        count = 0
        for y in y_cords:
            for x in x_cords:
                crop_img = image_bgr[y: y + tile_size, x: x + tile_size]
                crop_mask_layers = final_mask[y: y + tile_size, x: x + tile_size]

                # Картинку сохраняем как обычно
                img_save_path = os.path.join(dirmask, f"crop_{count}.png")
                cv2.imwrite(img_save_path, crop_img)

                # --- РАСКРАШИВАЕМ 2D МАСКУ В ЦВЕТНУЮ КАРТИНКУ ---
                h_crop, w_crop = crop_mask_layers.shape[:2]
                color_mask = numpy.zeros((h_crop, w_crop, 3), dtype=numpy.uint8)

                # Значение == 1: Руда -> серый (128, 128, 128)
                color_mask[crop_mask_layers == 1] = [128, 128, 128]

                # Значение == 2: Тальк -> белый (255, 255, 255)
                color_mask[crop_mask_layers == 2] = [255, 255, 255]

                # Значение == 3: Границы/срастания -> синий BGR (255, 0, 0)
                color_mask[crop_mask_layers == 3] = [255, 0, 0]

                # Сохраняем маску как нормальный видимый PNG
                mask_save_path = os.path.join(dirmask, f"mask_{count}.png")
                cv2.imwrite(mask_save_path, color_mask)
                count += 1
